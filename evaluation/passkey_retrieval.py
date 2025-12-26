import re
import os
import sys
import json
import itertools
import argparse
from dataclasses import dataclass, field, asdict
from typing import Any

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.helper import prepare_model, process_prompt, generate_response_with_params, inject_token_at_placeholder
from tokenizer.bpe_undertrained_norm_tokenizer import BPEUndertrainedNormTokenizer
from tokenizer.bpe_undertrained_entropy_tokenizer import BPEUndertrainedEntropyTokenizer
from quantifier.trainness.magikarp import TokenNorm
from quantifier.trainness.entropy import TokenEntropy

PROMPT_TEMPLATES = {
    "direct": "The secret password for the database is 'TOKEN'. Extract the password and spell it by placing a hyphen between every letter.\nOutput ONLY the hyphenated string. Do not provide any introductory text, labels, or explanations.",
    "narrative": "I was walking down the street and saw a sign that said 'TOKEN' in big neon letters.\nExtract the word written on the sign and spell the word with a hyphen between every letter. Constraint: Output ONLY the final hyphenated string. No other words.",
    "technical": "Extract the value of 'id' from this object: {{'id': 'TOKEN', 'status': 'active'}}. Extract the value of 'id' and spell it by placing a hyphen between every letter.\nOutput ONLY the hyphenated string. Do not provide any introductory text, labels, or explanations.",
    "long_context": "Please listen carefully. The weather is nice. The sky is blue. The target code is 'TOKEN'. The grass is green. Extract the target code and spell it by placing a hyphen between every letter.\nOutput ONLY the hyphenated string. Do not provide any introductory text, labels, or explanations.",
    "formatting": "The system generated a unique identifier: 'TOKEN'. Extract the identifier code and spell it by placing a hyphen between every letter.\nOutput ONLY the hyphenated string. Do not provide any introductory text, labels, or explanations."
}

PLACEHOLDER_TEXT = 'TOKEN'


@dataclass
class EvaluationEntry:
    input_ids: list[int]
    target_token_ids: list[int]
    target_token: str
    prompt_type: str
    input_tokens : list[str]
    response: str = ""
    error_char_rate: float = 0.0


@dataclass
class EvaluationStats:
    mean_error_char_rate: float
    total_errors: int
    total_chars: int
    total_count: int


@dataclass
class Dataset:
    entries: list[EvaluationEntry] = field(default_factory=list)

    def add_entry(self, input_ids: list[int], target_token_ids: list[int], 
                  target_token: str, prompt_type: str, input_tokens: list[str]) -> None:
        self.entries.append(EvaluationEntry(
            input_ids=input_ids,
            target_token_ids=target_token_ids,
            target_token=target_token,
            prompt_type=prompt_type,
            input_tokens=input_tokens
        ))

    def __len__(self) -> int:
        return len(self.entries)


def prepare_dataset(token_norm: TokenNorm | TokenEntropy, prompts: dict[str, Any], 
                    tokenizer, number_of_data: int = 500) -> Dataset:
    placeholder_ids = tokenizer.encode(PLACEHOLDER_TEXT, add_special_tokens=False)
    undertrained_tokens = token_norm.get_selected_undertrained_tokens(threshold='strong_verified')
    undertrained_tokens = dict(itertools.islice(undertrained_tokens.items(), number_of_data))

    dataset = Dataset()

    for prompt_type, prompt in prompts.items():
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)

        for data in undertrained_tokens.values():
            token_ids = tokenizer.encode(data['decoded'], add_special_tokens=False)
            try:
                new_input_ids = inject_token_at_placeholder(input_ids, placeholder_ids, token_ids)
                input_tokens = tokenizer.convert_ids_to_tokens(new_input_ids)
                dataset.add_entry(new_input_ids, token_ids, data['decoded'], prompt_type, input_tokens)
            except ValueError:
                continue

    return dataset


def parse_hyphenated(text: str) -> str:
    """Parse hyphenated string back to original (e.g., 'h-e-l-l-o' -> 'hello')."""
    text = text.strip()
    if '-' in text:
        return ''.join(text.split('-'))
    return text


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_error_char_rate(response: str, target_token: str) -> tuple[float, int, int]:
    """
    Calculate Error Character Rate (ECR) between response and target.
    
    Parses the hyphenated response and compares against the target token.
    
    Returns:
        tuple: (error_char_rate, num_errors, max_length)
    """
    target_token = target_token.strip()
    parsed_response = parse_hyphenated(response)
    
    edit_distance = levenshtein_distance(parsed_response, target_token)
    max_length = max(len(parsed_response), len(target_token))
    
    if max_length == 0:
        return 0.0, 0, 0
    
    error_rate = edit_distance / max_length
    return error_rate, edit_distance, max_length


def evaluate_responses(dataset: Dataset, responses: list[str]) -> EvaluationStats:
    total_errors = 0
    total_chars = 0
    sum_error_rates = 0.0

    for entry, response in zip(dataset.entries, responses):
        entry.response = response
        error_rate, errors, max_len = calculate_error_char_rate(response, entry.target_token)
        entry.error_char_rate = error_rate
        total_errors += errors
        total_chars += max_len
        sum_error_rates += error_rate

    total_count = len(dataset)
    mean_ecr = sum_error_rates / total_count if total_count > 0 else 0.0

    return EvaluationStats(
        mean_error_char_rate=mean_ecr,
        total_errors=total_errors,
        total_chars=total_chars,
        total_count=total_count
    )


def save_results(dataset: Dataset, output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset.entries:
            f.write(json.dumps(asdict(entry)) + '\n')
    print(f"Detailed results saved to {output_path}")


def save_stats(stats: EvaluationStats, output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(stats), f, indent=2)
    print(f"Stats saved to {output_path}")


def build_prompts(tokenizer, use_vllm: bool) -> dict[str, Any]:
    processed = process_prompt(tokenizer, list(PROMPT_TEMPLATES.values()), use_vllm)
    return {key: processed[i] for i, key in enumerate(PROMPT_TEMPLATES.keys())}


def main(args: argparse.Namespace) -> None:
    model, tokenizer = prepare_model(args.model_name, args.use_vllm)

    quantifier = None
    if args.quantifier_type == "norm":
        quantifier = TokenNorm(args.magikarp_path, tokenizer)
    elif args.quantifier_type == "entropy":
        quantifier = TokenEntropy(args.entropy_path, tokenizer, args.entropy_pkl)

    if args.tokenizer_type == 'norm':
        tokenizer = BPEUndertrainedNormTokenizer(tokenizer, quantifier, threshold='strong_verified')
    elif args.tokenizer_type == "entropy":
        tokenizer = BPEUndertrainedEntropyTokenizer(tokenizer, quantifier)

    prompts = build_prompts(tokenizer, args.use_vllm)
    dataset = prepare_dataset(quantifier, prompts, tokenizer, args.number_of_data)

    input_ids_list = [entry.input_ids for entry in dataset.entries]
    responses = generate_response_with_params(
        model, input_ids_list, tokenizer, args.use_vllm,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed
    )

    stats = evaluate_responses(dataset, responses)

    print("Evaluation Results:")
    print(f"Mean Error Character Rate: {stats.mean_error_char_rate * 100:.2f}%")
    print(f"Total Errors: {stats.total_errors} / {stats.total_chars} characters")
    print(f"Total Samples: {stats.total_count}")

    if args.output_path:
        save_results(dataset, args.output_path)
    
    if args.stats_path:
        save_stats(stats, args.stats_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Passkey Retrieval with Undertrained Tokens")
    parser.add_argument('--model_name', type=str, required=True, help='Pretrained model name or path')
    parser.add_argument('--magikarp_path', type=str, default=None, help='Path to Magikarp JSONL file')
    parser.add_argument('--entropy_path', type=str, default=None, help='Path to Entropy JSON file')
    parser.add_argument('--entropy_pkl', type=str, default=None, help='Path to Entropy pickle file')
    parser.add_argument('--quantifier_type', type=str, choices=['norm', 'entropy'], required=True, help='Type of quantifier to use')
    parser.add_argument('--tokenizer_type', type=str, choices=['standard', 'norm', 'entropy'], default='standard')
    parser.add_argument('--use_vllm', action='store_true', help='Whether to use vLLM for inference')
    parser.add_argument('--number_of_data', type=int, default=500, help='Number of undertrained tokens to evaluate')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0, help='Nucleus sampling probability')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save detailed results')
    parser.add_argument('--stats_path', type=str, default=None, help='Path to save stats JSON file')
    args = parser.parse_args()
    main(args)