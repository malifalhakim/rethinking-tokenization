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

from tqdm import tqdm
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

def initialize_seed(seed):
    """Initializes random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def setup_model_and_tokenizer(model_name, device_arg=None):
    """Loads the model and tokenizer and sets the device."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if device_arg:
        if device_arg.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device_arg)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer

def build_prompt(question, choices, mmlu_name, contamination_type='semantic'):
    """Builds the multiple-choice question prompt."""
    if "contaminated" in mmlu_name and contamination_type == "context":
        undertrained_word = question.split('--', 1)[0].rstrip()
        question = question.split('--', 1)[1].lstrip()
        prompt_text = f"{undertrained_word}\n\nQuestion: {question}\n"
    else:
        prompt_text = f"Question: {question}\n"
    for j, choice_text in enumerate(choices):
        prompt_text += f"{chr(65+j)}. {choice_text}\n"
    prompt_text += "Answer: "
    return prompt_text

def initialize_random_tokenizer(tokenizer, type:str="default", calculator:TokenNorm|TokenEntropy=None):
    """Initializes the random tokenizer."""
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

def get_choice_tokens(tokenizer, num_choices=4):
    """Gets the token IDs for the first few uppercase letters (A, B, C, ...)."""
    choice_tokens = {}
    for i in range(num_choices):
        char = chr(65 + i)
        
        token_id_no_space = tokenizer.encode(char, add_special_tokens=False)
        if token_id_no_space and len(token_id_no_space) == 1:
            choice_tokens[char] = token_id_no_space[0]
            continue
        
        token_id_with_space = tokenizer.encode(" " + char, add_special_tokens=False)
        if token_id_with_space and len(token_id_with_space) == 1:
            choice_tokens[char] = token_id_with_space[0]

    if len(choice_tokens) < num_choices:
        print(f"Warning: Could not find unique single-token representations for all choices A-{chr(65+num_choices-1)}")

    return choice_tokens

def evaluate_single_variant_by_prob(model, input_tensor, choice_token_ids):
    """
    Runs model inference and returns the choice with the highest probability.
    """
    model.eval()
    
    target_device = next(model.parameters()).device
    input_tensor = input_tensor.to(target_device)

    with torch.no_grad():
        outputs = model(input_tensor)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

    best_choice_char = None
    max_prob = -np.inf

    for char, token_id in choice_token_ids.items():
        prob = next_token_probs[0, token_id].item()
        if prob > max_prob:
            max_prob = prob
            best_choice_char = char

    return best_choice_char, np.round(max_prob, 4)

def get_input_variants(prompt_text, tokenizer, n=10):
    """Generates a list of input tensors based on the chosen tokenization strategy."""
    input_variants = []

    if isinstance(tokenizer, BPEAlternativeTokenizer):
        base_tokenizer = tokenizer.tokenizer
        encoded_inputs = tokenizer.encode(prompt_text, n=n, return_tensors="pt", add_special_tokens=True)
        for encoded_input in encoded_inputs:
            input_variants.append({
                "tensor": encoded_input, "desc": "alternative_tokenizer",
                "tokens_for_log": base_tokenizer.convert_ids_to_tokens(encoded_input[0])
            })
    else:
        encoded_input = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=True)
        input_variants.append({
            "tensor": encoded_input, "desc": "original_tokenizer",
            "tokens_for_log": tokenizer.convert_ids_to_tokens(encoded_input[0])
        })

    return input_variants

def evaluate(args):
    initialize_seed(args.seed)
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.device)
    if args.use_alternative_tokenizer:
        calculator = None
        if args.type == "norm" or args.type == "u-norm":
            file_path = args.quantifier_file
            if not file_path:
                raise ValueError("Quantifier file must be provided for 'norm' tokenizer type.")
            calculator = TokenNorm(file_path, tokenizer)
        elif args.type == "entropy" or args.type == "u-entropy":
            file_path = args.quantifier_file
            if not file_path:
                raise ValueError("Quantifier file must be provided for 'entropy' tokenizer type.")
            calculator = TokenEntropy(file_path, tokenizer)
        random_tokenizer = initialize_random_tokenizer(tokenizer, args.type, calculator=calculator)

    if args.subject == "all":
        try:
            dataset_info = load_dataset(args.mmlu_name, "all")
            subjects = list(dataset_info["test"].unique("subject"))
        except Exception as e:
            print(f"Could not load 'all' subjects configuration for cais/mmlu: {e}")
            subjects = []
    else:
        subjects = [args.subject]

    stats = {
        "model_name": args.model_name, "subject_accuracies": {}, "overall_accuracy": 0.0,
        "per_candidate_results": [], "num_samples_per_subject": args.num_samples or "all"
    }
    total_correct, total_evaluations = 0, 0

    dataset_all = load_dataset(args.mmlu_name, "all", split="test")
    for subject in subjects:
        print(f"Evaluating subject: {subject}")
        try:
            dataset = dataset_all.filter(lambda x: x["subject"] == subject)
            if args.num_samples:
                dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        except Exception as e:
            print(f"Could not load dataset for subject {subject}: {e}. Skipping.")
            stats["subject_accuracies"][subject] = f"N/A (dataset load error: {e})"
            continue

        if not dataset:
            stats["subject_accuracies"][subject] = "N/A (0 samples)"
            continue

        subject_correct = 0

        for i, item in enumerate(tqdm(dataset, desc=f"Subject {subject}")):
            prompt_text = build_prompt(item["question"], item["choices"], args.mmlu_name, args.contamination_type)
            actual_answer_index = item["answer"]
            actual_answer_char = chr(65 + int(actual_answer_index))

            choice_token_ids = get_choice_tokens(tokenizer, len(item["choices"]))
            if not choice_token_ids:
                print(f"  Q{i+1} ({subject}) - Skipping, could not get choice tokens.")
                continue
            
            if args.use_alternative_tokenizer:
                input_variants = get_input_variants(prompt_text, random_tokenizer, args.num_tokenizations_samples)
            else:
                input_variants = get_input_variants(prompt_text, tokenizer)

            question_answered_correctly = False
            for variant in input_variants:
                predicted_char, prob = evaluate_single_variant_by_prob(model, variant["tensor"], choice_token_ids)
                is_correct = (predicted_char == actual_answer_char)
                if is_correct:
                    question_answered_correctly = True

                print(f"  Q{i+1} ({subject}) - {variant['desc']} - Predicted: {predicted_char}, Actual: {actual_answer_char}, Correct: {is_correct}")
                stats["per_candidate_results"].append({
                    "subject": subject, "question_index": i + 1, "candidate_description": variant['desc'],
                    "predicted_char": predicted_char, "correct_char": actual_answer_char, "is_correct": is_correct,
                    "tokens_used": variant['tokens_for_log'],
                    "probability": prob
                })

            if question_answered_correctly:
                subject_correct += 1
        
        if len(dataset) > 0:
            subject_accuracy = subject_correct / len(dataset)
            stats["subject_accuracies"][subject] = f"{subject_accuracy:.4f} ({subject_correct}/{len(dataset)})"
            print(f"Accuracy for {subject}: {subject_accuracy:.4f}")

        total_correct += subject_correct
        total_evaluations += len(dataset)

        del dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if total_evaluations > 0:
        overall_accuracy = total_correct / total_evaluations
        stats["overall_accuracy"] = f"{overall_accuracy:.4f} ({total_correct}/{total_evaluations})"
        print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    
    safe_model_name = args.model_name.replace('/', '_')
    safe_mmlu_name = args.mmlu_name.split('/')[-1]
    output_filename = f"{args.contamination_type}_{safe_mmlu_name}_evaluation_stats_{safe_model_name}.json"
    if args.use_alternative_tokenizer:
        output_filename = f"{args.contamination_type}_{safe_mmlu_name}_evaluation_stats_{safe_model_name}_altok_{args.type}.json"
    with open(output_filename, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Evaluation statistics saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MMLU model performance")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate")
    parser.add_argument("--subject", type=str, default="all", help="Subject to evaluate (default: all)")
    parser.add_argument("--mmlu_name", type=str, default="cais/mmlu", help="Hugging Face dataset name for MMLU (default: cais/mmlu)")
    parser.add_argument("--contamination_type", type=str, default="semantic", choices=["semantic", "context"], help="Type of contamination in the dataset (default: semantic)")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--use_alternative_tokenizer", action="store_true", help="Use random tokenizer for generating alternatives")
    parser.add_argument("--type", type=str, default="default", choices=["default", "filtered", "norm", "entropy", "renyi", "u-norm", "u-entropy"], help="Type of random tokenizer to use (default or filtered)")
    parser.add_argument("--num_tokenizations_samples", type=int, default=8, help="Number of alternative tokenizations to generate (default: 8)")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cuda:0, cuda:1, cpu. If omitted uses device_map=auto.")
    parser.add_argument("--quantifier_file", type=str, default=None, help="Path to quantifier file for norm/entropy tokenizers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    evaluate(args)