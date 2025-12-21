import os
import sys
import argparse
import json
import re
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from quantifier.trainness.magikarp import TokenNorm
from tokenizer.bpe_norm_tokenizer import BPENormTokenizer
from utils.helper import prepare_model, process_prompt, inject_token_at_placeholder, generate_response_with_params

PROMPT_TEMPLATE = """Answer the following math problem. 
You must think step-by-step to solve it.
At the very end of your response, write the final numerical answer starting with "####".

Question:
{problem_text}

Answer:"""

PLACEHOLDER_TEXT = 'TOKEN'


def load_jsonl(file_path: str) -> pd.DataFrame:
    """Load a JSONL file into a pandas DataFrame."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    if not data:
        raise ValueError(f"Dataset at {file_path} is empty.")

    return pd.DataFrame(data)


def apply_prompt(tokenizer, problems: list[str], use_vllm: bool) -> list[str]:
    """Format problems with the prompt template."""
    prompts = [PROMPT_TEMPLATE.format(problem_text=problem) for problem in problems]
    return process_prompt(tokenizer, prompts, use_vllm)


def preprocess_dataset(dataset: pd.DataFrame, tokenizer, token_norm: TokenNorm) -> pd.DataFrame:
    """
    Preprocess dataset by injecting undertrained tokens at placeholders.
    
    Args:
        dataset: DataFrame with 'formatted_question', 'problem', and 'answer' columns
        tokenizer: Tokenizer to encode text
        token_norm: TokenNorm instance for getting undertrained tokens
        
    Returns:
        DataFrame with processed inputs including injected tokens
    """
    placeholder_ids = tokenizer.encode(PLACEHOLDER_TEXT, add_special_tokens=False)
    undertrained_tokens = token_norm.get_selected_undertrained_tokens(threshold='strong_verified')
    undertrained_tokens_list = list(undertrained_tokens.values())

    if not undertrained_tokens_list:
        raise ValueError("No undertrained tokens found with the given threshold.")

    pointer = 0
    processed_inputs = []
    for _, row in dataset.iterrows():
        formatted_question = row['formatted_question']
        input_ids = tokenizer.encode(formatted_question, add_special_tokens=False)
        undertrained_token = undertrained_tokens_list[pointer % len(undertrained_tokens_list)]['decoded']
        token_ids = tokenizer.encode(undertrained_token, add_special_tokens=False)

        try:
            new_input_ids = inject_token_at_placeholder(input_ids, placeholder_ids, token_ids)
            processed_inputs.append({
                'problem': row['problem'],
                'answer': row['answer'],
                'input_ids': new_input_ids,
                'target_token_ids': token_ids,
                'target_token': undertrained_token,
                'input_tokens': tokenizer.convert_ids_to_tokens(new_input_ids)
            })
            pointer += 1
        except ValueError:
            continue

    if not processed_inputs:
        raise ValueError("No samples were successfully processed. Check placeholder injection.")

    return pd.DataFrame(processed_inputs)


def extract_answer(response: str) -> str:
    """Extract the final answer after #### delimiter."""
    delimiter = '####'
    if delimiter not in response:
        return ""
    parts = response.split(delimiter)
    return parts[-1].strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (remove commas, extra spaces, handle numeric formats)."""
    # Remove commas from numbers (e.g., "1,000" -> "1000")
    normalized = re.sub(r',', '', answer)
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    # Try to normalize numeric format
    try:
        num = float(normalized)
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return normalized


def check_answer_correct(predicted: str, expected: str) -> bool:
    """Check if predicted answer matches expected answer after normalization."""
    return normalize_answer(predicted) == normalize_answer(expected)


def save_stats(accuracy: float, total_samples: int, correct_samples: int, output_path: str) -> None:
    """Save evaluation statistics to JSON file."""
    stats = {
        'accuracy': accuracy,
        'total_samples': total_samples,
        'correct_samples': correct_samples
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)


def save_detailed_results(dataset: pd.DataFrame, output_path: str) -> None:
    """Save detailed results to JSONL file."""
    dataset.to_json(output_path, orient='records', lines=True)


def main(args: argparse.Namespace) -> None:
    """Main evaluation pipeline."""
    dataset = load_jsonl(args.dataset_path)

    if 'problem' not in dataset.columns or 'answer' not in dataset.columns:
        raise ValueError("Dataset must contain 'problem' and 'answer' columns.")
    
    if args.limit is not None and args.limit > 0:
        dataset = dataset.head(args.limit)

    model, tokenizer = prepare_model(args.model_name, args.use_vllm)
    
    token_norm = TokenNorm(args.magikarp_path, tokenizer)
    if args.tokenizer_type == 'norm':
        tokenizer = BPENormTokenizer(tokenizer, token_norm)

    dataset['formatted_question'] = apply_prompt(tokenizer, dataset['problem'].tolist(), args.use_vllm)
    dataset = preprocess_dataset(dataset, tokenizer, token_norm)

    input_ids_list = dataset['input_ids'].tolist()
    responses = generate_response_with_params(
        model, input_ids_list, tokenizer, args.use_vllm,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed
    )

    dataset['response'] = responses
    dataset['predicted_answer'] = dataset['response'].apply(extract_answer)
    dataset['is_correct'] = dataset.apply(
        lambda row: check_answer_correct(row['predicted_answer'], row['answer']), 
        axis=1
    )

    correct_count = dataset['is_correct'].sum()
    total_count = len(dataset)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct_count}/{total_count})")

    save_stats(accuracy, total_count, int(correct_count), args.stats_output_path)
    save_detailed_results(dataset, args.detailed_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K with undertrained token injection.")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the GSM8K dataset JSONL file.')
    parser.add_argument('--magikarp_path', type=str, required=True, help='Path to the Magikarp JSONL file.')
    parser.add_argument('--model_name', type=str, required=True, help='Name or path of the pre-trained model.')
    parser.add_argument('--use_vllm', action='store_true', help='Whether to use vLLM for inference.')
    parser.add_argument('--tokenizer_type', type=str, choices=['standard', 'norm'], default='standard', help='Type of tokenizer to use.')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to evaluate.')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum number of new tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature.')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p sampling parameter.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation.')
    parser.add_argument('--stats_output_path', type=str, required=True, help='Path to save evaluation stats JSON file.')
    parser.add_argument('--detailed_output_path', type=str, required=True, help='Path to save detailed results JSONL file.')

    args = parser.parse_args()
    main(args)


