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

def build_prompt(review_text):
    """Builds the sentiment classification prompt."""
    return f'Classify the sentiment of the following Amazon review. Only answer with the word "Positive" or "Neutral" or "Negative".\n\nReview: "{review_text}"\n\nSentiment: '

def initialize_alternative_tokenizer(tokenizer, type:str, calculator:TokenNorm|TokenEntropy=None):
    """Initializes the alternative tokenizer."""
    if type == "norm":
        return BPEUndertrainedNormTokenizer(tokenizer, token_norm=calculator)
    elif type == "entropy":
        return BPEUndertrainedEntropyTokenizer(tokenizer, token_entropy=calculator)
    return None

def get_sentiment_tokens(tokenizer):
    """Gets the token IDs for the sentiment classes (Positive, Neutral, Negative)."""
    sentiment_tokens = {}
    for sentiment in ["Positive", "Neutral", "Negative"]:
        token_id = tokenizer.encode(sentiment, add_special_tokens=False)
        if token_id and len(token_id) == 1:
            sentiment_tokens[sentiment] = token_id[0]
            continue

        token_id_space = tokenizer.encode(" " + sentiment, add_special_tokens=False)
        if token_id_space and len(token_id_space) == 1:
            sentiment_tokens[sentiment] = token_id_space[0]

    if len(sentiment_tokens) != len(["Positive", "Neutral", "Negative"]):
        print("Warning: Could not find unique single-token representations for all sentiment classes.")

    return sentiment_tokens

def evaluate_single_variant_by_prob(model, input_tensor, sentiment_token_ids):
    """
    Runs model inference and returns the sentiment with the highest probability.
    """
    model.eval()
    
    target_device = next(model.parameters()).device
    input_tensor = input_tensor.to(target_device)

    try:
        with torch.no_grad():
            outputs = model(input_tensor)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)

        best_choice_char = None
        max_prob = -np.inf

        for sentiment, token_id in sentiment_token_ids.items():
            prob = next_token_probs[0, token_id].item()
            if prob > max_prob:
                max_prob = prob
                best_choice_char = sentiment

        return best_choice_char, np.round(max_prob, 4)
    except torch.cuda.OutOfMemoryError:
        print("CUDA Out of Memory during evaluation. Skipping this input.")
        torch.cuda.empty_cache()
        return None, None

def get_input_variants(prompt_text, tokenizer, n=1):
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
        alt_tokenizer = initialize_alternative_tokenizer(tokenizer, args.type, calculator=calculator)

    if args.category == "all":
        try:
            categories = [
                "All_Beauty",
                "Amazon_Fashion",
                "Appliances",
                "Arts_Crafts_and_Sewing",
                "Automotive",
                "Baby_Products",
                "Beauty_and_Personal_Care",
                "Books",
                "CDs_and_Vinyl",
                "Cell_Phones_and_Accessories",
                "Clothing_Shoes_and_Jewelry",
                "Digital_Music",
                "Electronics",
                "Gift_Cards",
                "Grocery_and_Gourmet_Food",
                "Handmade_Products",
                "Health_and_Household",
                "Health_and_Personal_Care",
                "Home_and_Kitchen",
                "Industrial_and_Scientific",
                "Kindle_Store",
                "Magazine_Subscriptions",
                "Movies_and_TV",
                "Musical_Instruments",
                "Office_Products",
                "Patio_Lawn_and_Garden",
                "Pet_Supplies",
                "Software",
                "Sports_and_Outdoors",
                "Subscription_Boxes",
                "Tools_and_Home_Improvement",
                "Toys_and_Games",
                "Video_Games",
                "Unknown"
            ]
        except Exception as e:
            print(f"Could not load 'all' categories configuration for {args.dataset_name}: {e}")
            categories = []
    else:
        categories = [args.category]

    stats = {
        "model_name": args.model_name, "category_accuracies": {}, "overall_accuracy": 0.0,
        "per_candidate_results": [], "num_samples_per_category": args.num_samples or "all"
    }
    total_correct, total_evaluations = 0, 0

    for category in categories:
        print(f"Evaluating category: {category}")
        try:
            dataset = load_dataset(args.dataset_name, category, split="test")
            if args.num_samples:
                dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        except Exception as e:
            print(f"Could not load dataset for category {category}: {e}. Skipping.")
            stats["category_accuracies"][category] = f"N/A (dataset load error: {e})"
            continue

        if not dataset:
            stats["category_accuracies"][category] = "N/A (0 samples)"
            continue

        category_correct = 0
        sentiment_token_ids = get_sentiment_tokens(tokenizer)

        for i, item in enumerate(tqdm(dataset, desc=f"Category {category}")):
            prompt_text = build_prompt(item["review"])
            actual_answer_sentiment = item["sentiment"]

            if not sentiment_token_ids:
                print(f"  Q{i+1} ({category}) - Skipping, could not get sentiment tokens.")
                continue
            
            if args.use_alternative_tokenizer:
                input_variants = get_input_variants(prompt_text, alt_tokenizer, args.num_tokenizations_samples)
            else:
                input_variants = get_input_variants(prompt_text, tokenizer)

            question_answered_correctly = False
            for variant in input_variants:
                predicted_sentiment, prob = evaluate_single_variant_by_prob(model, variant["tensor"], sentiment_token_ids)
                is_correct = (predicted_sentiment == actual_answer_sentiment)
                if is_correct:
                    question_answered_correctly = True

                print(f"  Q{i+1} ({category}) - {variant['desc']} - Predicted: {predicted_sentiment}, Actual: {actual_answer_sentiment}, Correct: {is_correct}")
                stats["per_candidate_results"].append({
                    "category": category, "question_index": i + 1, "candidate_description": variant['desc'],
                    "predicted_sentiment": predicted_sentiment, "correct_sentiment": actual_answer_sentiment, "is_correct": is_correct,
                    "tokens_used": variant['tokens_for_log'],
                    "probability": prob
                })

            if question_answered_correctly:
                category_correct += 1
        
        if len(dataset) > 0:
            category_accuracy = category_correct / len(dataset)
            stats["category_accuracies"][category] = f"{category_accuracy:.4f} ({category_correct}/{len(dataset)})"
            print(f"Accuracy for {category}: {category_accuracy:.4f}")

        total_correct += category_correct
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
    output_filename = f"amazon_evaluation_stats_{safe_model_name}.json"
    if args.use_alternative_tokenizer:
        output_filename = f"amazon_evaluation_stats_{safe_model_name}_altok_{args.type}.json"
    with open(output_filename, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Evaluation statistics saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Amazon Reviews model performance")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--use_alternative_tokenizer", action="store_true", help="Use random tokenizer for generating alternatives")
    parser.add_argument("--type", type=str, default="norm", choices=["norm", "entropy", "renyi"], help="Type of random tokenizer to use (default or filtered)")
    parser.add_argument("--num_tokenizations_samples", type=int, default=1, help="Number of alternative tokenizations to generate (default: 8)")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cuda:0, cuda:1, cpu. If omitted uses device_map=auto.")
    parser.add_argument("--quantifier_file", type=str, default=None, help="Path to quantifier file for norm/entropy tokenizers.")
    parser.add_argument("--undertrained_entropy_file", type=str, default=None, help="Path to undertrained entropy pkl file for undertrained entropy tokenizer.")
    parser.add_argument("--dataset_name", type=str, default="amazon_reviews_multi", help="Name of the dataset to use (default: amazon_reviews_multi)")
    parser.add_argument("--category", type=str, default="all", help="Category to evaluate (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    evaluate(args)