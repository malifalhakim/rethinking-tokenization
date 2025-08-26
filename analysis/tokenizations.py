import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
import matplotlib.pyplot as plt

from quantifier.efficiency.renyi import renyi_score
from quantifier.trainness.entropy import TokenEntropy

from transformers import AutoTokenizer

def read_json(file_path: str) -> dict:
    """Reads a JSON file and returns its content as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_tokenizations_data(subject: str, question_index: int, raw_data: list, entropy_calculator: TokenEntropy) -> list:
    """Get tokenizations used for a specific subject & question_index."""
    tokenizations = []
    for detail in raw_data:
        if detail.get('question_index') == question_index and detail.get('subject') == subject:
            tokens_used = detail['tokens_used']
            tokenizations.append(
                {
                    "candidate_description": detail.get('candidate_description'),
                    "tokens_used": tokens_used,
                    "entropy": entropy_calculator.get_entropy_score(tokens_used),
                    "renyi": renyi_score(tokens_used),
                    "is_correct": detail.get('is_correct')
                }
            )
    return tokenizations

def plot_entropy_renyi_scatter(tokenizations_1: list, tokenizations_2: list,
                               label1: str = "Data 1", label2: str = "Data 2",
                               annotate: bool = False):
    """
    Scatter plot: x = entropy, y = renyi to compare tokenizations quantifier.
    """
    style_map = {
        ('data1', True):  {'color': "#09ef3f", 'marker': 'o', 'label': f'{label1} correct'},
        ('data1', False): {'color': "#d90202", 'marker': 'o', 'label': f'{label1} incorrect'},
        ('data2', True):  {'color': "#0cea52", 'marker': 's', 'label': f'{label2} correct'},
        ('data2', False): {'color': "#ea2d14", 'marker': 's', 'label': f'{label2} incorrect'},
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    plotted_labels = set()

    def plot_group(data, tag):
        for item in data:
            key = (tag, bool(item['is_correct']))
            style = style_map[key]
            lbl = style['label'] if style['label'] not in plotted_labels else None
            ax.scatter(item['entropy'], item['renyi'],
                       c=style['color'], marker=style['marker'],
                       edgecolors='black', linewidths=0.6, s=70, alpha=0.85,
                       label=lbl)
            if lbl:
                plotted_labels.add(style['label'])
            if annotate:
                ax.text(item['entropy'], item['renyi'],
                        item.get('candidate_description', ''),
                        fontsize=7, alpha=0.7,
                        ha='left', va='bottom')

    plot_group(tokenizations_1, 'data1')
    plot_group(tokenizations_2, 'data2')

    ax.set_xlabel('Entropy', fontsize=12)
    ax.set_ylabel('Rényi score', fontsize=12)
    ax.set_title('Entropy vs Rényi Tokenization Scatter', fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()

def main(args):
    raw_data_1 = read_json(args.input_1)["per_candidate_results"]
    raw_data_2 = read_json(args.input_2)["per_candidate_results"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    entropy_calculator = TokenEntropy(args.entropy_file, tokenizer)

    tokenizations_1 = get_tokenizations_data(
        subject=args.subject, question_index=args.question_index, raw_data=raw_data_1, entropy_calculator=entropy_calculator)
    tokenizations_2 = get_tokenizations_data(
        subject=args.subject, question_index=args.question_index, raw_data=raw_data_2, entropy_calculator=entropy_calculator)

    plot_entropy_renyi_scatter(
        tokenizations_1,
        tokenizations_2,
        label1=os.path.basename(args.input_1),
        label2=os.path.basename(args.input_2),
        annotate=args.annotate
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare tokenizations for a specific question.")
    parser.add_argument("--input_1", required=True, help="Path to first JSON file.")
    parser.add_argument("--input_2", required=True, help="Path to second JSON file.")
    parser.add_argument("--subject", type=str, default='abstract_algebra', help="Subject name.")
    parser.add_argument("--question_index", type=int, default=1, help="Question index.")
    parser.add_argument("--annotate", action="store_true", help="Annotate points with candidate descriptions.")
    parser.add_argument('--entropy_file', type=str, required=True, help="Path to token entropy JSON file.")
    parser.add_argument('--tokenizer_name', type=str, required=True, help="Tokenizer name or path.")
    args = parser.parse_args()
    main(args)