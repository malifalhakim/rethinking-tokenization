import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import random
import gc
import argparse
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate as evaluate_hf

from tokenizer.bpe_random_tokenizer import BPEAlternativeTokenizer
from tokenizer.bpe_undertrained_norm_tokenizer import BPEUndertrainedNormTokenizer
from tokenizer.bpe_undertrained_entropy_tokenizer import BPEUndertrainedEntropyTokenizer

from quantifier.trainness.magikarp import TokenNorm
from quantifier.trainness.entropy import TokenEntropy

def initialize_seed(seed: int):
    """Initializes random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def setup_model_and_tokenizer(model_name, device_arg=None):
    """Loads the model and tokenizer and sets the device."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device_arg:
        if device_arg.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device_arg)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer

def build_translation_prompt(source_text, src_lang, tgt_lang):
    """Builds a prompt for the translation task."""
    map_langcode_to_name = {
        "EN": "English",
        "FR": "French",
        "JA": "Japanese",
    }

    prompt_text = f"Translate the following text from {map_langcode_to_name.get(src_lang, src_lang)} to {map_langcode_to_name.get(tgt_lang, tgt_lang)}. Provide only the translated text.\n\n{src_lang}: {source_text}\n{tgt_lang}:"
    messages = [
        {"role": "user", "content": prompt_text}
    ]

    return messages

def initialize_alternative_tokenizer(tokenizer, type: str, calculator: TokenNorm|TokenEntropy=None):
    """
    Initializes your custom random tokenizer, imitating the mmlu.py structure.
    """
    if type == "norm":
        return BPEUndertrainedNormTokenizer(tokenizer, token_norm=calculator)
    elif type == "entropy":
        return BPEUndertrainedEntropyTokenizer(tokenizer, token_entropy=calculator)
    return None

def get_input_variants(prompt_text, tokenizer, n=1):
    """
    Generates a list of input tensors based on the chosen tokenization strategy.
    This structure is kept to match your original script.
    """
    input_variants = []
    is_custom_tokenizer = isinstance(tokenizer, (BPEAlternativeTokenizer))

    if is_custom_tokenizer:
        base_tokenizer = tokenizer.tokenizer
        formatted_prompt = base_tokenizer.apply_chat_template(prompt_text, tokenize=False, add_generation_prompt=True)

        encoded_inputs = tokenizer.encode(formatted_prompt, n=n, return_tensors="pt", add_special_tokens=True)
        for i, encoded_input in enumerate(encoded_inputs):
            input_variants.append({
                "tensor": encoded_input, "desc": f"alternative_tokenizer_variant_{i+1}",
                "tokens_for_log": base_tokenizer.convert_ids_to_tokens(encoded_input[0])
            })
    else:
        encoded_input_ids = tokenizer.apply_chat_template(prompt_text, return_tensors="pt", add_generation_prompt=True)
        input_variants.append({
            "tensor": encoded_input_ids, "desc": "original_tokenizer",
            "tokens_for_log": tokenizer.convert_ids_to_tokens(encoded_input_ids[0])
        })

    return input_variants

def generate_translation(model, tokenizer, input_tensor, max_new_tokens=128):
    """
    Runs model inference to generate a translation.
    """
    model.eval()
    target_device = next(model.parameters()).device
    input_tensor = input_tensor.to(target_device)

    with torch.no_grad():
        outputs = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    generated_text = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

def evaluate(args):
    """Main evaluation function."""
    initialize_seed(args.seed)
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.device)
    sacrebleu = evaluate_hf.load("sacrebleu")
    
    eval_tokenizer = tokenizer
    if args.use_alternative_tokenizer:
        calculator = None
        if args.type == "norm":
            file_path = args.quantifier_file
            if not file_path:
                raise ValueError("Quantifier file must be provided for 'norm' tokenizer type.")
            calculator = TokenNorm(file_path, tokenizer)
        elif args.type == "entropy":
            file_path = args.quantifier_file
            pkl_file = args.undertrained_entropy_file
            if not file_path:
                raise ValueError("Quantifier file must be provided for 'entropy' tokenizer type.")
            calculator = TokenEntropy(file_path, tokenizer, pkl_file_path=pkl_file)
        eval_tokenizer = initialize_alternative_tokenizer(tokenizer, args.type, calculator=calculator)

    lang_pair = args.dataset_path.split(".")[1]
    src_lang, tgt_lang = lang_pair.split('-')

    stats = {
        "model_name": args.model_name,
        "dataset": args.dataset_path,
        "lang_pair": lang_pair,
        "num_samples": args.num_samples or "all",
        "overall_score": {},
        "sample_translations": []
    }

    print(f"Evaluating on local dataset: {args.dataset_path}")
    try:
        dataset = pd.read_csv(args.dataset_path)
        if args.num_samples:
            dataset = dataset.head(min(args.num_samples, len(dataset)))
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {args.dataset_path}.")
        return
    except Exception as e:
        print(f"Could not load dataset from {args.dataset_path}: {e}.")
        return

    predictions = []
    references = []

    for i, row in enumerate(tqdm(dataset.itertuples(), total=len(dataset), desc=f"Translating {lang_pair}")):
        source_text = row.original_text
        reference_text = row.translated_text

        prompt_text = build_translation_prompt(source_text, src_lang.upper(), tgt_lang.upper())
        input_variants = get_input_variants(prompt_text, eval_tokenizer, args.num_tokenizations_samples)
        
        best_score = -1.0
        best_prediction = ""
        for variant in input_variants:
            predicted_text = generate_translation(model, tokenizer, variant["tensor"], args.max_new_tokens)
            score_result = sacrebleu.compute(predictions=[predicted_text], references=[[reference_text]])
            current_score = score_result["score"]
            if current_score > best_score:
                best_score = current_score
                best_prediction = predicted_text

            stats["sample_translations"].append({
                "dataset_index": i,
                "candidate_description": variant.get("desc", "unknown"),
                "predicted_text": predicted_text,
                "score": current_score,
                "tokens_used": variant.get("tokens_for_log", [])
            })

        predictions.append(best_prediction)
        references.append([reference_text])

        print(f"\n--- Sample {i+1} ---\nPrediction: {best_prediction}\nScore: {best_score}\nTokens: {variant.get('tokens_for_log', [])}")

    if predictions:
        results = sacrebleu.compute(predictions=predictions, references=references)
        stats["overall_score"] = results
        print("\n--- Evaluation Results ---")
        print(json.dumps(results, indent=2))
    else:
        print("No samples were evaluated.")
    
    safe_model_name = args.model_name.replace('/', '_')
    output_filename = f"mtnt_evaluation_stats_{safe_model_name}_{lang_pair}.json"
    if args.use_alternative_tokenizer:
        output_filename = f"mtnt_evaluation_stats_{safe_model_name}_{lang_pair}_altok_{args.type}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    print(f"\nEvaluation statistics saved to {output_filename}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Machine Translation model performance on a local CSV.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Hugging Face model to evaluate.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the local CSV dataset file. Must contain 'original_text' and 'translated_text' columns.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (default: all).")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate for each translation.")
    parser.add_argument("--use_alternative_tokenizer", action="store_true", help="Use your custom alternative tokenizer for generating alternatives.")
    parser.add_argument("--type", type=str, default="norm", choices=["norm", "entropy","renyi"], help="Type of alternative tokenizer to use (if applicable).")
    parser.add_argument("--num_tokenizations_samples", type=int, default=1, help="Number of alternative tokenizations to generate")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. 'cuda:0', 'cpu'. If omitted, uses 'device_map=auto'.")
    parser.add_argument("--quantifier_file", type=str, default=None, help="Path to the quantifier file (required for 'norm' and 'entropy' types).")
    parser.add_argument("--undertrained_entropy_file", type=str, default=None, help="Path to undertrained entropy pkl file for undertrained entropy tokenizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    evaluate(args)

