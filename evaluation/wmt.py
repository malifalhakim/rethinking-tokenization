import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gc
import json
import torch
import argparse
import pandas as pd
import evaluate as evaluate_hf

from tqdm import tqdm
from typing import Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tokenizer.bpe_random_tokenizer import BPEAlternativeTokenizer
from tokenizer.bpe_random_tokenizer_filtered import BPEAlternativeTokenizerFiltered
from tokenizer.bpe_norm_tokenizer import BPENormTokenizer
from tokenizer.bpe_entropy_tokenizer import BPEEntropyTokenizer
from tokenizer.bpe_renyi_tokenizer import BPERenyiTokenizer
from tokenizer.bpe_undertrained_norm_tokenizer import BPEUndertrainedNormTokenizer
from tokenizer.bpe_undertrained_entropy_tokenizer import BPEUndertrainedEntropyTokenizer

from quantifier.trainness.magikarp import TokenNorm
from quantifier.trainness.entropy import TokenEntropy

def initialize_seed(seed: int):
    """Initializes random seeds for reproducibility."""
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def prepare_dataset(wmt_name: str = "wmt14", dataset_subset: str = "fr-en", src_lang: str = "en") -> tuple:
    """
    Loads and prepares the WMT14 dataset.
    """
    langs = set(dataset_subset.split('-'))
    if src_lang not in langs:
        raise ValueError(f"Source language '{src_lang}' is not in the subset '{dataset_subset}'.")

    tgt_lang = (langs - {src_lang}).pop()

    dataset = load_dataset(wmt_name, dataset_subset, split="test")
    return dataset, tgt_lang

def process_dataset(dataset, src_lang: str, tgt_lang: str, num_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Processes a dataset object into a pandas DataFrame.
    """
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    translations = dataset['translation']
    return pd.DataFrame({
        "original_text": [item[src_lang] for item in translations],
        "translated_text": [item[tgt_lang] for item in translations]
    })

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

def build_translation_prompt(source_text, src_lang, tgt_lang, wmt_name, contamination_type='semantic'):
    """Builds a prompt for the translation task."""
    map_langcode_to_name = {
        "EN": "English",
        "FR": "French",
        "DE": "German",
        "CS": "Czech",
        "RU": "Russian",
        "HI": "Hindi",
    }

    if "contaminated" in wmt_name and contamination_type == "context":
        undertrained_word = source_text.split('--', 1)[0].rstrip()
        source_text = source_text.split('--', 1)[1].lstrip()
        prompt_text = f"{undertrained_word}\n\nTranslate the following text from {map_langcode_to_name.get(src_lang, src_lang)} to {map_langcode_to_name.get(tgt_lang, tgt_lang)}. Provide only the translated text.\n\n{src_lang}: {source_text}\n{tgt_lang}:"
    else:
        prompt_text = f"Translate the following text from {map_langcode_to_name.get(src_lang, src_lang)} to {map_langcode_to_name.get(tgt_lang, tgt_lang)}. Provide only the translated text.\n\n{src_lang}: {source_text}\n{tgt_lang}:"
    messages = [
        {"role": "user", "content": prompt_text}
    ]

    return messages

def initialize_random_tokenizer(tokenizer, type: str = "filtered", calculator: TokenNorm|TokenEntropy=None):
    """
    Initializes your custom random tokenizer, imitating the mmlu.py structure.
    """
    if type == "filtered":
        return BPEAlternativeTokenizerFiltered(tokenizer)
    elif type == "norm":
        return BPENormTokenizer(tokenizer, token_norm=calculator)
    elif type == "entropy":
        return BPEEntropyTokenizer(tokenizer, token_entropy=calculator)
    elif type == "renyi":
        return BPERenyiTokenizer(tokenizer)
    elif type == "u-norm":
        return BPEUndertrainedNormTokenizer(tokenizer, token_norm=calculator, threshold="strong_verified")
    elif type == "u-entropy":
        return BPEUndertrainedEntropyTokenizer(tokenizer, token_entropy=calculator)
    return BPEAlternativeTokenizer(tokenizer)

def get_input_variants(messages, tokenizer, n=10):
    """
    Generates a list of input tensors based on the chosen tokenization strategy.
    This structure is kept to match your original script.
    """
    input_variants = []
    is_custom_tokenizer = isinstance(tokenizer, (BPEAlternativeTokenizer))

    if is_custom_tokenizer:
        base_tokenizer = tokenizer.tokenizer
        formatted_prompt = base_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        encoded_inputs = tokenizer.encode(formatted_prompt, n=n, return_tensors="pt", add_special_tokens=True)
        for i, encoded_input in enumerate(encoded_inputs):
            input_variants.append({
                "tensor": encoded_input, "desc": f"alternative_tokenizer_variant_{i+1}",
                "tokens_for_log": base_tokenizer.convert_ids_to_tokens(encoded_input[0])
            })
    else:
        encoded_input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
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
    
    input_tokenizer = tokenizer
    if args.use_alternative_tokenizer:
        calculator = None
        if args.type == "norm" or args.type == "u-norm":
            file_path = args.quantifier_file
            calculator = TokenNorm(file_path, tokenizer)
        elif args.type == "entropy" or args.type == "u-entropy":
            file_path = args.quantifier_file
            pkl_path = args.entropy_pkl_file
            calculator = TokenEntropy(file_path, tokenizer, pkl_file_path=pkl_path)
        input_tokenizer = initialize_random_tokenizer(tokenizer, args.type, calculator=calculator)

    dataset_subset = args.dataset_subset
    src_lang = args.src_lang

    dataset, tgt_lang = prepare_dataset(args.wmt_name, dataset_subset, src_lang)
    dataset = process_dataset(dataset, src_lang, tgt_lang, args.num_samples)
    lang_pair = f"{src_lang}-{tgt_lang}"

    stats = {
        "model_name": args.model_name,
        "dataset": args.dataset_subset,
        "lang_pair": lang_pair,
        "num_samples": args.num_samples or "all",
        "overall_score": {},
        "sample_translations": []
    }

    predictions = []
    references = []

    for i, row in enumerate(tqdm(dataset.itertuples(), total=len(dataset), desc=f"Translating {lang_pair}")):
        source_text = row.original_text
        reference_text = row.translated_text

        messages = build_translation_prompt(source_text, src_lang.upper(), tgt_lang.upper(), args.wmt_name, args.contamination_type)
        input_variants = get_input_variants(messages, input_tokenizer, args.num_tokenizations_samples)
        
        best_score = -1.0
        best_prediction = ""
        for variant in input_variants:
            predicted_text = generate_translation(model, tokenizer, variant["tensor"], args.max_new_tokens)
            
            problem = False
            if "contaminated" in args.wmt_name and args.contamination_type == "context":
                try:
                    reference_text = reference_text.split('--',1)[1].lstrip()
                except IndexError:
                    problem = True
                    pass
            
            if "contaminated" in args.wmt_name and args.contamination_type == "semantic":
                try:
                    reference_text = reference_text.split('--',1)[1].lstrip()
                    predicted_text = predicted_text.split('--',1)[1].lstrip()
                except IndexError:
                    try:
                        reference_1 = reference_text.split('--',1)[1].lstrip()
                        predicted_1 = predicted_text
                        score_result_1 = sacrebleu.compute(predictions=[predicted_1], references=[[reference_1]])

                        reference_2 = f"{source_text.split('--',1)[0].rstrip()} -- {reference_text.split('--',1)[1].lstrip()}"
                        predicted_2 = predicted_text
                        score_result_2 = sacrebleu.compute(predictions=[predicted_2], references=[[reference_2]])

                        if score_result_1["score"] >= score_result_2["score"]:
                            reference_text = reference_1
                            predicted_text = predicted_1
                        else:
                            reference_text = reference_2
                            predicted_text = predicted_2
                    except IndexError:
                        problem = True
                        print(f"Reference: {reference_text}")
                        print(f"Predicted: {predicted_text}")
                        print(f"Warning: Could not parse contaminated format for sample {i}")
                        pass
                except Exception as e:
                    problem = True
                    print(f"Reference: {reference_text}")
                    print(f"Predicted: {predicted_text}")
                    print(f"Warning: Error processing contaminated sample {i}: {e}")
                    pass

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
                "tokens_used": variant.get("tokens_for_log", []),
                "problem": problem
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
    wmt_name_clean = args.wmt_name.split('/')[-1]
    output_filename = f"{args.contamination_type}_{wmt_name_clean}_evaluation_stats_{safe_model_name}_{lang_pair}.json"
    if args.use_alternative_tokenizer:
        output_filename = f"{args.contamination_type}_{wmt_name_clean}_evaluation_stats_{safe_model_name}_{lang_pair}_altok_{args.type}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    print(f"\nEvaluation statistics saved to {output_filename}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a language model on WMT translation tasks.")
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name or path.")
    parser.add_argument("--wmt_name", type=str, default="wmt14", help="WMT dataset name (default: wmt14).")
    parser.add_argument("--contamination_type", type=str, default="semantic", choices=["semantic", "context"], help="Type of contamination in the dataset (default: semantic)")
    parser.add_argument("--dataset_subset", type=str, default="fr-en", help="WMT dataset subset (e.g., 'fr-en').")
    parser.add_argument("--src_lang", type=str, default="en", help="Source language code (e.g., 'en').")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate. Default is all.")
    parser.add_argument("--device", type=str, default=None, help="Device to run the model on (e.g., 'cuda', 'cpu').")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate.")
    parser.add_argument("--use_alternative_tokenizer", action="store_true", help="Use a random tokenizer variant.")
    parser.add_argument("--type", type=str, choices=["standard", "filtered", "norm", "entropy", "renyi", "u-norm", "u-entropy"], default="filtered", help="Type of random tokenizer to use.")
    parser.add_argument("--num_tokenizations_samples", type=int, default=4, help="Number of tokenization samples for random tokenizer.")
    parser.add_argument("--quantifier_file", type=str, default=None, help="Path to the quantifier file for norm/entropy tokenizers.")
    parser.add_argument("--entropy_pkl_file", type=str, default=None, help="Path to entropy pkl file for entropy-based tokenizers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    try:
        evaluate(args)
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")