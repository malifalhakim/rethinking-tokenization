import os
import sys
import argparse
import json
import torch
import gc
from datetime import datetime
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import load_dataset
from quantifier.trainness.magikarp import TokenNorm
from tokenizer.bpe_norm_tokenizer import BPENormTokenizer
from utils.helper import prepare_model, process_prompt, find_optimal_batch_size

PROMPT_TEMPLATE = """{question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
Answer with the option letter.
ANSWER: """

OPTION_LABELS = ["A", "B", "C", "D"]


def apply_prompt(question: str, options: list[str]) -> str:
    """Format question and options into the prompt template."""
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}")
    
    return PROMPT_TEMPLATE.format(
        question=question,
        option_a=options[0],
        option_b=options[1],
        option_c=options[2],
        option_d=options[3],
    )


def get_option_token_ids(tokenizer) -> dict[str, int]:
    """Get token IDs for option labels A, B, C, D."""
    option_tokens = {}
    for label in OPTION_LABELS:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Could not encode option label '{label}'")
        option_tokens[label] = token_ids[0]
    return option_tokens


def get_model_probabilities(model, tokenized_prompts: dict) -> torch.Tensor:
    """Get model output probabilities for the tokenized prompts."""
    with torch.no_grad():
        outputs = model(**tokenized_prompts)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
    return probabilities


def get_option_probabilities(
    probabilities: torch.Tensor, 
    tokenized_prompts: dict, 
    option_tokens: dict[str, int]
) -> list[dict[str, float]]:
    """Extract probabilities for option tokens from the last position."""
    attention_mask = tokenized_prompts["attention_mask"]
    seq_lengths = attention_mask.sum(dim=1) - 1
    
    batch_size = probabilities.shape[0]
    option_probs = []
    
    for i in range(batch_size):
        last_pos = seq_lengths[i].item()
        last_token_probs = probabilities[i, last_pos, :]
        
        probs = {
            label: last_token_probs[token_id].item()
            for label, token_id in option_tokens.items()
        }
        option_probs.append(probs)
    
    return option_probs


def evaluate_subject(
    model, 
    tokenizer, 
    subject_data, 
    option_tokens: dict[str, int],
    batch_size: int,
    collect_tokenized: bool = False
) -> tuple[list[dict], int, int]:
    """Evaluate model on a single subject's data."""
    prompts = [
        apply_prompt(item["question"], item["choices"]) 
        for item in subject_data
    ]
    
    all_option_probs = []
    all_tokenized_strings = [] if collect_tokenized else None
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokenized_prompts = process_prompt(tokenizer, batch_prompts, use_vllm=False)
        
        if collect_tokenized:
            for j in range(tokenized_prompts["input_ids"].shape[0]):
                tokens_as_strings = tokenizer.convert_ids_to_tokens(
                    tokenized_prompts["input_ids"][j].tolist()
                )
                all_tokenized_strings.append(tokens_as_strings)
        
        tokenized_prompts = {k: v.to(model.device) for k, v in tokenized_prompts.items()}
        
        probabilities = get_model_probabilities(model, tokenized_prompts)
        batch_probs = get_option_probabilities(probabilities, tokenized_prompts, option_tokens)
        all_option_probs.extend(batch_probs)
        
        # Clean up GPU memory
        del tokenized_prompts, probabilities
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate results
    logs = []
    correct = 0
    
    for idx, (item, probs) in enumerate(zip(subject_data, all_option_probs)):
        predicted = max(probs, key=probs.get)
        correct_answer = OPTION_LABELS[item["answer"]]
        
        if predicted == correct_answer:
            correct += 1
        
        log_entry = {
            "question": item["question"],
            "predicted": predicted,
            "correct": correct_answer,
            "is_correct": predicted == correct_answer,
            "probs": probs,
        }
        
        if collect_tokenized and all_tokenized_strings:
            log_entry["tokenized_input"] = all_tokenized_strings[idx]
        
        logs.append(log_entry)
    
    return logs, correct, len(subject_data)


def save_results(logs: list, stats: dict, output_path: str):
    """Save evaluation results to a JSON file."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "statistics": stats,
        "detailed_logs": logs,
    }
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")


def main(args):
    # -- Load dataset --
    print("Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    
    if args.limit is not None and args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited dataset to {len(dataset)} samples")
    
    subjects = dataset.unique("subject")
    
    # -- Prepare model and tokenizer --
    print(f"Loading model: {args.model_name}")
    model, tokenizer = prepare_model(args.model_name, use_vllm=False)

    # -- Get option token IDs --
    option_tokens = get_option_token_ids(tokenizer)
    
    # -- Apply custom tokenizer if specified --
    if args.tokenizer_type == "norm":
        if not args.magikarp_path:
            raise ValueError("--magikarp_path is required when using --tokenizer_type norm")
        print(f"Loading TokenNorm from: {args.magikarp_path}")
        token_norm = TokenNorm(args.magikarp_path, tokenizer)
        tokenizer = BPENormTokenizer(tokenizer, token_norm)
    
    # -- Find optimal batch size --
    sample_data = dataset.select(range(min(10, len(dataset))))
    sample_prompts = [
        apply_prompt(item["question"], item["choices"]) 
        for item in sample_data
    ]
    
    optimal_batch_size = find_optimal_batch_size(
        model, tokenizer, sample_prompts, 
        use_vllm=False, max_new_tokens=1, start_batch_size=args.batch_size
    )
    print(f"Using batch size: {optimal_batch_size}")
    
    # -- Evaluate each subject --
    all_logs = []
    stats = {}
    total_correct = 0
    total_count = 0
    
    for subject in tqdm(subjects, desc="Evaluating subjects"):
        subject_data = dataset.filter(lambda x: x["subject"] == subject)
        
        logs, correct, count = evaluate_subject(
            model, tokenizer, subject_data, option_tokens,
            batch_size=optimal_batch_size,
            collect_tokenized=args.save_tokenized
        )
        
        # -- Add subject to each log entry --
        for log in logs:
            log["subject"] = subject
        
        all_logs.extend(logs)
        
        accuracy = correct / count if count > 0 else 0.0
        stats[subject] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": count
        }
        
        total_correct += correct
        total_count += count
        
        print(f"Subject: {subject}, Accuracy: {accuracy:.4f} ({correct}/{count})")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # -- Calculate overall accuracy --
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0
    stats["_overall"] = {
        "accuracy": overall_accuracy,
        "correct": total_correct,
        "total": total_count
    }
    
    print(f"\n{'='*50}")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_count})")
    print(f"{'='*50}")
    
    # -- Save results --
    if args.output_path:
        save_results(all_logs, stats, args.output_path)
    
    return stats, all_logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU dataset")
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--tokenizer_type", 
        type=str, 
        choices=["norm", "standard"], 
        default=None, 
        help="Type of tokenizer to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--magikarp_path", 
        type=str, 
        default=None, 
        help="Path to Magikarp JSONL file for token normalization"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save evaluation results (JSON)"
    )
    parser.add_argument(
        "--save_tokenized",
        action="store_true",
        help="Save tokenized inputs in the logs"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples in the dataset"
    )
    
    args = parser.parse_args()
    main(args)