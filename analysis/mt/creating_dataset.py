import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import argparse
import pandas as pd

from quantifier.efficiency.renyi import renyi_score
from quantifier.trainness.entropy import TokenEntropy
from quantifier.trainness.magikarp import TokenNorm

from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer

def read_json(file_path: str) -> dict:
    """Reads a JSON file and returns its content as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_tokenizations_data(raw_data: list, entropy_calculator: TokenEntropy, norm_calculator: TokenNorm) -> list:
    """Get tokenizations used for WMT translation data."""
    tokenizations = []
    for detail in raw_data:
        tokens_used = detail['tokens_used']
        tokenizations.append(
            {
                "dataset_index": detail.get('dataset_index'),
                "candidate_description": detail.get('candidate_description'),
                "predicted_text": detail.get('predicted_text'),
                "tokens_used": tokens_used,
                "entropy": entropy_calculator.get_score(tokens_used),
                "l2_norm": norm_calculator.get_score(tokens_used),
                "renyi": renyi_score(tokens_used),
                "score": detail.get('score')
            }
        )
    return tokenizations

def initialize_calculators(tokenizer_name: str, entropy_file: str, magikarp_file: str):
    """Initialize entropy and norm calculators."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    entropy_calculator = TokenEntropy(entropy_file, tokenizer)
    norm_calculator = TokenNorm(magikarp_file, tokenizer)
    return entropy_calculator, norm_calculator

def create_dataset_list(tokenizer_name: str, entropy_file: str, magikarp_file: str, input_1: str, input_2: str) -> list:
    """Create dataset comparing baseline and alternative tokenizations for MT data."""
    entropy_calculator, norm_calculator = initialize_calculators(tokenizer_name, entropy_file, magikarp_file)

    baseline_data = read_json(input_1)
    alternative_data = read_json(input_2)
    
    raw_baseline_tokenizations = baseline_data["sample_translations"]
    raw_alternative_tokenizations = alternative_data["sample_translations"]

    baseline_tokenizations = get_tokenizations_data(raw_baseline_tokenizations, entropy_calculator, norm_calculator)
    alternative_tokenizations = get_tokenizations_data(raw_alternative_tokenizations, entropy_calculator, norm_calculator)

    data_dict = []
    for baseline_tokenization in tqdm(baseline_tokenizations):
        dataset_index = baseline_tokenization['dataset_index']
        baseline_score = baseline_tokenization['score']
        
        for alternative_tokenization in alternative_tokenizations:
            dataset_index_alt = alternative_tokenization['dataset_index']
            alternative_score = alternative_tokenization['score']
            
            if dataset_index == dataset_index_alt:
                if alternative_score > baseline_score:
                    label = 1  # Alternative tokenization is better
                elif alternative_score < baseline_score:
                    label = 0  # Baseline tokenization is better
                else:
                    label = -1  # Scores are equal (tie)
                
                data_dict.append({
                    "dataset_index": dataset_index,
                    "baseline_score": baseline_score,
                    "alternative_score": alternative_score,
                    "score_difference": alternative_score - baseline_score,
                    "renyi_baseline": baseline_tokenization['renyi'],
                    "renyi_alternative": alternative_tokenization['renyi'],
                    "norm_baseline": baseline_tokenization['l2_norm'],
                    "norm_alternative": alternative_tokenization['l2_norm'],
                    "entropy_baseline": baseline_tokenization['entropy'],
                    "entropy_alternative": alternative_tokenization['entropy'],
                    "tokens_baseline": baseline_tokenization['tokens_used'],
                    "tokens_alternative": alternative_tokenization['tokens_used'],
                    "predicted_text_baseline": baseline_tokenization['predicted_text'],
                    "predicted_text_alternative": alternative_tokenization['predicted_text'],
                    "label": label
                })

    return data_dict

def transform_to_pandas(data: List[dict]) -> pd.DataFrame:
    """Transforms the dataset list of dicts into a pandas DataFrame."""
    return pd.DataFrame(data)

def save_to_csv(dataframe: pd.DataFrame, output_path: str):
    """Save dataframe to defined output_path"""
    dataframe.to_csv(output_path, index=False)

def main(args):
    tokenizer_name = args.tokenizer_name
    entropy_file = args.entropy_file
    magikarp_file = args.magikarp_file
    input_1 = args.input_1
    input_2 = args.input_2
    output_path = args.output_path

    data = create_dataset_list(tokenizer_name, entropy_file, magikarp_file, input_1, input_2)
    df = transform_to_pandas(data)
    save_to_csv(df, output_path)
    
    print(f"Dataset created with {len(df)} samples")
    print(f"Label distribution:")
    print(df['label'].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ML dataset from WMT tokenization evaluation data.")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer to use.")
    parser.add_argument("--entropy_file", type=str, required=True, help="Path to the entropy file.")
    parser.add_argument("--magikarp_file", type=str, required=True, help="Path to the Magikarp file.")
    parser.add_argument("--input_1", type=str, required=True, help="Path to the baseline WMT evaluation JSON file.")
    parser.add_argument("--input_2", type=str, required=True, help="Path to the alternative WMT evaluation JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output CSV file.")

    args = parser.parse_args()
    main(args)