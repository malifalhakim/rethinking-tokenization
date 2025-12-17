import re
import os
import sys
import json
import itertools
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.helper import prepare_model, process_prompt, generate_response_with_params, inject_token_at_placeholder
from tokenizer.bpe_norm_tokenizer import BPENormTokenizer
from quantifier.trainness.magikarp import TokenNorm

prompt_templates = {
    "direct": "The secret password for the database is 'TOKEN'. What is the password?",
    "narrative": "I was walking down the street and saw a sign that said 'TOKEN' in big neon letters. What did the sign say?",
    "technical": "Extract the value of 'id' from this object: {{'id': 'TOKEN', 'status': 'active'}}. Return only the value.",
    "long_context": "Please listen carefully. The weather is nice. The sky is blue. The target code is 'TOKEN'. The grass is green. What is the target code?",
    "formatting": "The system generated a unique identifier: 'TOKEN'. Please output this identifier and nothing else."
}


def prepare_dataset(token_norm: TokenNorm, prompts: dict, tokenizer, number_of_data: int = 500):
    placeholder_text = 'TOKEN'
    placeholder_ids = tokenizer.encode(placeholder_text, add_special_tokens=False)

    undertrained_tokens = token_norm.get_selected_undertrained_tokens()
    undertrained_tokens = dict(itertools.islice(undertrained_tokens.items(), number_of_data))

    dataset_entries = {
        'input_ids': [],
        'target_token_ids': [],
        'target_token': [],
        'prompt_type': []
    }

    for prompt_type, prompt in prompts.items():
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)

        for _, data in undertrained_tokens.items():
            token_ids = tokenizer.encode(data['decoded'], add_special_tokens=False)

            try:
                new_input_ids = inject_token_at_placeholder(input_ids, placeholder_ids, token_ids)
                dataset_entries['input_ids'].append(new_input_ids)
                dataset_entries['target_token_ids'].append(token_ids)
                dataset_entries['target_token'].append(data['decoded'])
                dataset_entries['prompt_type'].append(prompt_type)
            except ValueError:
                continue

    return dataset_entries

def is_match(response: str, target_token: str) -> bool:
    target_token = target_token.strip()
    target_token = re.escape(target_token)

    pattern = rf"{target_token}"
    match = re.search(pattern, response)

    return bool(match)

def evaluate_responses(responses: list[str], target_tokens: list[str]) -> dict:
    correct_count = 0
    recorded_correctness = []
    total_count = len(responses)

    for response, target_token in zip(responses, target_tokens):
        if is_match(response, target_token):
            correct_count += 1
            recorded_correctness.append(True)
        else:
            recorded_correctness.append(False)

    accuracy = correct_count / total_count if total_count > 0 else 0.0
    stats = {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': total_count
    }

    return stats, recorded_correctness


def main(args):
    model, tokenizer = prepare_model(args.model_name, args.use_vllm)

    token_norm = TokenNorm(args.magikarp_path, tokenizer)

    if args.tokenizer_type == 'norm':
        tokenizer = BPENormTokenizer(tokenizer, token_norm)
    
    processed_prompts = process_prompt(tokenizer, list(prompt_templates.values()), args.use_vllm)
    prompts = {key: processed_prompts[i] for i, key in enumerate(prompt_templates.keys())}

    dataset = prepare_dataset(token_norm, prompts, tokenizer, args.number_of_data)
    responses = generate_response_with_params(
        model, 
        dataset['input_ids'], 
        tokenizer, 
        args.use_vllm, 
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed
    )

    stats, correctness = evaluate_responses(responses, dataset['target_token'])

    dataset['responses'] = responses
    dataset['correctness'] = correctness

    print("Evaluation Results:")
    print(f"Accuracy: {stats['accuracy']*100:.2f}% ({stats['correct_count']}/{stats['total_count']})")
    if args.output_path:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            for i in range(len(dataset['input_ids'])):
                entry = {
                    'input_ids': dataset['input_ids'][i],
                    'target_token_ids': dataset['target_token_ids'][i],
                    'target_token': dataset['target_token'][i],
                    'prompt_type': dataset['prompt_type'][i],
                    'response': dataset['responses'][i],
                    'correctness': dataset['correctness'][i]
                }
                f.write(json.dumps(entry) + '\n')
        print(f"Detailed results saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Passkey Retrieval with Undertrained Tokens")
    parser.add_argument('--model_name', type=str, required=True, help='Pretrained model name or path')
    parser.add_argument('--magikarp_path', type=str, required=True, help='Path to Magikarp JSONL file')
    parser.add_argument('--tokenizer_type', type=str, choices=['standard', 'norm'], default='standard', help='Type of tokenizer to use')
    parser.add_argument('--use_vllm', action='store_true', help='Whether to use vLLM for inference')
    parser.add_argument('--number_of_data', type=int, default=500, help='Number of undertrained tokens to evaluate')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature for generation')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p sampling value for generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save detailed evaluation results')

    args = parser.parse_args()
    main(args)