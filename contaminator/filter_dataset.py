import os
import json
import pickle
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from huggingface_hub import HfApi

def parse_arguments():
    """Parse command line arguments for the dataset filtering script."""
    parser = argparse.ArgumentParser(
        description="Filter a dataset to include only samples containing undertrained tokens."
    )
    
    parser.add_argument("--source-dataset", type=str, default="HPLT/HPLT2.0_cleaned",
                       help="Name of the source dataset to filter")
    parser.add_argument("--magikarp-path", type=str,
                       help="Path to magikarp output file (JSONL format)")
    parser.add_argument("--glitchminer-path", type=str,
                       help="Path to glitchminer output file (pickle format)")
    parser.add_argument("--output-dataset", type=str, default="Amadeus/hplt_with_undertrained_tokens",
                       help="Name for the output dataset on Hugging Face Hub")
    parser.add_argument("--max-samples", type=int, default=10000,
                       help="Maximum number of samples to collect")
    parser.add_argument("--text-column", type=str, default="text",
                       help="Name of the text column in the dataset")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to filter")
    parser.add_argument("--tokenizer-name", type=str, required=True,
                       help="Name or path of the tokenizer to use for tokenization")
    parser.add_argument("--private", action="store_true",
                       help="Create a private repository")
    
    return parser.parse_args()

def load_magikarp_tokens(file_path):
    """Load undertrained tokens from magikarp JSONL output file."""
    tokens = []
    file_path = Path(file_path)
    
    with file_path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                if data.get('magikarp') == 'strong_verified' and data.get('category') == 'OK':
                    if decoded := data.get('decoded'):
                        tokens.append(decoded)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON at line {line_num}")
    
    return tokens

def load_glitchminer_tokens(file_path):
    """Load undertrained tokens from glitchminer pickle output file."""
    file_path = Path(file_path)
    
    with file_path.open('rb') as f:
        data = pickle.load(f)
    
    return data.get('glitch_tokens', [])

def load_undertrained_tokens(magikarp_path=None, glitchminer_path=None):
    """Load undertrained tokens from either magikarp or glitchminer file."""
    if magikarp_path:
        print(f"Loading tokens from magikarp file: '{magikarp_path}'")
        tokens = load_magikarp_tokens(magikarp_path)
    elif glitchminer_path:
        print(f"Loading tokens from glitchminer file: '{glitchminer_path}'")
        tokens = load_glitchminer_tokens(glitchminer_path)
    else:
        raise ValueError("Provide either --magikarp-path or --glitchminer-path")
    
    print(f"Loaded {len(tokens)} undertrained tokens")
    return set(tokens)

def load_tokenizer(tokenizer_name):
    """Load the specified tokenizer from Hugging Face."""
    print(f"Loading tokenizer: {tokenizer_name}")
    return AutoTokenizer.from_pretrained(tokenizer_name)

def stream_dataset(source_dataset, split):
    """Stream the specified dataset split for memory-efficient processing."""
    print(f"Streaming dataset: '{source_dataset}'")
    return load_dataset(source_dataset, split=split, streaming=True)

def check_sample_contains_tokens(text, tokenizer, undertrained_tokens_set):
    """Check if text contains any undertrained tokens after tokenization."""
    tokens = tokenizer.tokenize(text)
    token_set = set(tokens)
    return bool(undertrained_tokens_set & token_set)

def filter_samples(dataset, tokenizer, undertrained_tokens_set, max_samples, text_column):
    """Filter dataset samples that contain undertrained tokens."""
    filtered_samples = []
    count = 0
    processed_count = 0

    for example in tqdm(dataset, desc="Scanning samples"):
        if count >= max_samples:
            break
        
        processed_count += 1
        
        try:
            text = example[text_column]
            if check_sample_contains_tokens(text, tokenizer, undertrained_tokens_set):
                filtered_samples.append(example)
                count += 1
                if count % 100 == 0:
                    print(f"Collected {count}/{max_samples} samples (processed {processed_count})")
        except Exception as e:
            print(f"Warning: Skipping sample due to error: {e}")
            continue

    print(f"Finished filtering: {len(filtered_samples)} samples from {processed_count} processed")
    return filtered_samples

def create_repository(repo_id, private=False):
    """Create a Hugging Face dataset repository if it doesn't exist."""
    api = HfApi()
    
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"Repository '{repo_id}' exists")
        return True
    except Exception:
        print(f"Creating repository: '{repo_id}'")
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"Repository '{repo_id}' created")
        return True

def push_dataset_to_hub(samples, output_dataset, private=False):
    """Convert samples to dataset and push to Hugging Face Hub."""
    dataset = Dataset.from_list(samples)
    
    if not create_repository(output_dataset, private):
        raise Exception("Failed to create repository")
    
    print(f"Pushing dataset to Hub: '{output_dataset}'")
    dataset.push_to_hub(output_dataset, private=private)
    print("Successfully pushed dataset to Hub")

def display_configuration(args):
    """Display the current configuration settings."""
    print("Configuration:")
    print(f"  Source dataset: {args.source_dataset}")
    print(f"  Magikarp path: {args.magikarp_path}")
    print(f"  Glitchminer path: {args.glitchminer_path}")
    print(f"  Output dataset: {args.output_dataset}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Text column: {args.text_column}")
    print(f"  Split: {args.split}")
    print(f"  Tokenizer: {args.tokenizer_name}")
    print(f"  Private: {args.private}")
    print("-" * 50)

def main():
    """Main function to orchestrate the dataset filtering process."""
    args = parse_arguments()
    display_configuration(args)
    
    try:
        undertrained_tokens_set = load_undertrained_tokens(args.magikarp_path, args.glitchminer_path)
        tokenizer = load_tokenizer(args.tokenizer_name)
        dataset = stream_dataset(args.source_dataset, args.split)
        
        filtered_samples = filter_samples(
            dataset, tokenizer, undertrained_tokens_set, 
            args.max_samples, args.text_column
        )
        
        if filtered_samples:
            push_dataset_to_hub(filtered_samples, args.output_dataset, args.private)
        else:
            print("No samples collected")
            
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()