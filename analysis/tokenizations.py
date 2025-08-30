import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
import matplotlib.pyplot as plt
from typing import List

try:
    import mplcursors
except ImportError:
    mplcursors = None

from quantifier.efficiency.renyi import renyi_score
from quantifier.trainness.entropy import TokenEntropy
from quantifier.trainness.magikarp import TokenNorm
from tokenizer.bpe_random_tokenizer_filtered import is_random_bpe

from transformers import AutoTokenizer

def read_json(file_path: str) -> dict:
    """Reads a JSON file and returns its content as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_tokenizations_data(subject: str, question_index: int, raw_data: list, calculator: TokenEntropy|TokenNorm) -> list:
    """Get tokenizations used for a specific subject & question_index."""
    tokenizations = []
    for detail in raw_data:
        if detail.get('question_index') == question_index and detail.get('subject') == subject:
            tokens_used = detail['tokens_used']
            tokenizations.append(
                {
                    "candidate_description": detail.get('candidate_description'),
                    "tokens_used": tokens_used,
                    "trainness_score": calculator.get_score(tokens_used),
                    "renyi": renyi_score(tokens_used),
                    "is_correct": detail.get('is_correct')
                }
            )
    return tokenizations

def filter_tokenizations_data(alternative_tokenizations: List[dict], canonical_tokenization: List[str]) -> List[dict]:
    """Filter alternative tokenizations to only those that are not random BPE of canonical tokenizations."""
    filtered = []
    for alt in alternative_tokenizations:
        if not is_random_bpe(alt['tokens_used'], canonical_tokenization):
            filtered.append(alt)
    return filtered

def plot_entropy_renyi_scatter(tokenizations_1: list, tokenizations_2: list,
                               label1: str = "Data 1", label2: str = "Data 2",
                               annotate: bool = False,
                               hover: bool = False):
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
    scatter_artists = []
    descriptions = []

    def plot_group(data, tag):
        for item in data:
            key = (tag, bool(item['is_correct']))
            style = style_map[key]
            lbl = style['label'] if style['label'] not in plotted_labels else None
            sc = ax.scatter(item['trainness_score'], item['renyi'],
                            c=style['color'], marker=style['marker'],
                            edgecolors='black', linewidths=0.6, s=70, alpha=0.85,
                            label=lbl)
            scatter_artists.append(sc)
            descriptions.append(item.get('tokens_used', ''))
            if lbl:
                plotted_labels.add(style['label'])
            if annotate:
                ax.text(item['trainness_score'], item['renyi'],
                        item.get('candidate_description', ''),
                        fontsize=7, alpha=0.7,
                        ha='left', va='bottom')

    plot_group(tokenizations_1, 'data1')
    plot_group(tokenizations_2, 'data2')

    ax.set_xlabel('trainness_score', fontsize=12)
    ax.set_ylabel('Rényi score', fontsize=12)
    ax.set_title('trainness_score vs Rényi Tokenization Scatter', fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(frameon=False, fontsize=9)

    if hover:
        if mplcursors is None:
            print("mplcursors not installed. Install with: pip install mplcursors")
        else:
            cursor = mplcursors.cursor(scatter_artists, hover=True)

            @cursor.connect("add")
            def on_add(sel):
                try:
                    idx = scatter_artists.index(sel.artist)
                except ValueError:
                    idx = -1
                text = descriptions[idx] if idx >= 0 else ""
                sel.annotation.set_text(text)
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9, edgecolor="#333")

    plt.tight_layout()
    plt.show()

def main(args):
    raw_data_1 = read_json(args.input_1)["per_candidate_results"]
    raw_data_2 = read_json(args.input_2)["per_candidate_results"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if args.entropy_file:
        calculator = TokenEntropy(args.entropy_file, tokenizer)
    elif args.magikarp_file:
        calculator = TokenNorm(args.magikarp_file, tokenizer)
    else:
        print("Error: Either --entropy_file or --magikarp_file must be provided.")

    tokenizations_1 = get_tokenizations_data(
        subject=args.subject, question_index=args.question_index, raw_data=raw_data_1, calculator=calculator)
    tokenizations_2 = get_tokenizations_data(
        subject=args.subject, question_index=args.question_index, raw_data=raw_data_2, calculator=calculator)
    
    if args.filter_random_bpe:
        tokenizations_2 = filter_tokenizations_data(tokenizations_2, tokenizations_1[0]['tokens_used'])

    plot_entropy_renyi_scatter(
        tokenizations_1,
        tokenizations_2,
        label1=os.path.basename(args.input_1),
        label2=os.path.basename(args.input_2),
        annotate=args.annotate,
        hover=args.hover
    )

    if args.run_diff:
        print("\n=== Running tokenization difference analysis ===\n")
        from tokenization_difference import main as diff_main
        diff_args = argparse.Namespace(
            input_1=args.input_1,
            input_2=args.input_2,
            subject=args.subject,
            question_index=args.question_index
        )
        diff_main(diff_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare tokenizations for a specific question.")
    parser.add_argument("--input_1", required=True, help="Path to first JSON file.")
    parser.add_argument("--input_2", required=True, help="Path to second JSON file.")
    parser.add_argument("--subject", type=str, default='abstract_algebra', help="Subject name.")
    parser.add_argument("--question_index", type=int, default=1, help="Question index.")
    parser.add_argument("--annotate", action="store_true", help="Annotate points with candidate descriptions.")
    parser.add_argument('--magikarp_file', type=str, help="Path to Magikarp JSONL file.")
    parser.add_argument('--entropy_file', type=str, help="Path to token entropy JSON file.")
    parser.add_argument('--tokenizer_name', type=str, required=True, help="Tokenizer name or path.")
    parser.add_argument('--filter_random_bpe', action="store_true", help="Filter out random BPE tokenizations.")
    parser.add_argument('--hover', action="store_true", help="Enable hover tooltips (requires mplcursors).")
    parser.add_argument('--run_diff', action="store_true", help="After plotting, run tokenization_difference diff output.")
    args = parser.parse_args()
    main(args)