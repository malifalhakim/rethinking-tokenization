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
from tokenizer.bpe_norm_tokenizer import BPENormTokenizer
from quantifier.trainness.magikarp import TokenNorm

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
    response: str = ""
    correctness: bool = False


@dataclass
class EvaluationStats:
    accuracy: float
    correct_count: int
    total_count: int


@dataclass
class Dataset:
    entries: list[EvaluationEntry] = field(default_factory=list)

    def add_entry(self, input_ids: list[int], target_token_ids: list[int], 
                  target_token: str, prompt_type: str) -> None:
        self.entries.append(EvaluationEntry(
            input_ids=input_ids,
            target_token_ids=target_token_ids,
            target_token=target_token,
            prompt_type=prompt_type
        ))

    def __len__(self) -> int:
        return len(self.entries)


def prepare_dataset(token_norm: TokenNorm, prompts: dict[str, Any], 
                    tokenizer, number_of_data: int = 500) -> Dataset:
    placeholder_ids = tokenizer.encode(PLACEHOLDER_TEXT, add_special_tokens=False)
    undertrained_tokens = token_norm.get_selected_undertrained_tokens()
    undertrained_tokens = dict(itertools.islice(undertrained_tokens.items(), number_of_data))

    dataset = Dataset()

    for prompt_type, prompt in prompts.items():
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)

        for data in undertrained_tokens.values():
            token_ids = tokenizer.encode(data['decoded'], add_special_tokens=False)
            try:
                new_input_ids = inject_token_at_placeholder(input_ids, placeholder_ids, token_ids)
                dataset.add_entry(new_input_ids, token_ids, data['decoded'], prompt_type)
            except ValueError:
                continue

    return dataset


def is_match(response: str, target_token: str) -> bool:
    target_token = target_token.strip()
    pattern = re.escape(target_token)
    return bool(re.search(pattern, response))


def evaluate_responses(dataset: Dataset, responses: list[str]) -> EvaluationStats:
    correct_count = 0

    for entry, response in zip(dataset.entries, responses):
        entry.response = response
        entry.correctness = is_match(response, entry.target_token)
        if entry.correctness:
            correct_count += 1

    total_count = len(dataset)
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    return EvaluationStats(accuracy=accuracy, correct_count=correct_count, total_count=total_count)


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
    token_norm = TokenNorm(args.magikarp_path, tokenizer)

    if args.tokenizer_type == 'norm':
        tokenizer = BPENormTokenizer(tokenizer, token_norm)

    prompts = build_prompts(tokenizer, args.use_vllm)
    dataset = prepare_dataset(token_norm, prompts, tokenizer, args.number_of_data)

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
    print(f"Accuracy: {stats.accuracy * 100:.2f}% ({stats.correct_count}/{stats.total_count})")

    if args.output_path:
        save_results(dataset, args.output_path)
    
    if args.stats_path:
        save_stats(stats, args.stats_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Passkey Retrieval with Undertrained Tokens")
    parser.add_argument('--model_name', type=str, required=True, help='Pretrained model name or path')
    parser.add_argument('--magikarp_path', type=str, required=True, help='Path to Magikarp JSONL file')
    parser.add_argument('--tokenizer_type', type=str, choices=['standard', 'norm'], default='standard')
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