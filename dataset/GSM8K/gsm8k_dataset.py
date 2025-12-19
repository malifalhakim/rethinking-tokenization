import argparse
import json

from datasets import load_dataset

def load_jsonl(file_path: str) -> list[dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def replace_words(text: str, target: str) -> str:
    PLACEHOLDER = "\'TOKEN\'"

    text = text.replace(target, PLACEHOLDER)
    if target.lower() in text:
        text = text.replace(target.lower(), PLACEHOLDER)
    
    return text

def extract_answer(solution: str) -> str:
    delimiter = '####'
    parts = solution.split(delimiter)
    return parts[-1].strip()

def main(args):
    infos = load_jsonl(args.dataset_path)
    dataset = load_dataset('openai/gsm8k', name='main', split='test')

    processed_data = []
    for info, entry in zip(infos, dataset):
        problem_text = entry['question']
        replaceable = info['is_safe']
        if not replaceable:
            continue
        
        target_words = info['target_word']
        problem_text = replace_words(problem_text, target_words)
        answer = extract_answer(entry['answer'])

        processed_data.append({
            'problem': problem_text,
            'answer': answer
        })

    print(f"Total processed entries: {len(processed_data)}")
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process GSM8K dataset with target word replacements.")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the JSONL file containing target word info.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the processed dataset.')
    args = parser.parse_args()
    main(args)

