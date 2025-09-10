import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
import matplotlib.pyplot as plt

try:
    import mplcursors
except ImportError:
    mplcursors = None

from quantifier.efficiency.renyi import renyi_score
from quantifier.trainness.entropy import TokenEntropy
from quantifier.trainness.magikarp import TokenNorm

from transformers import AutoTokenizer

def read_json(file_path: str) -> dict:
    """Reads a JSON file and returns its content as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_tokenizations_data(data_index: int, raw_data: list, calculator: TokenEntropy|TokenNorm) -> list:
    """Get tokenizations used for a specific data_index."""
    tokenizations = []
    for detail in raw_data:
        if detail.get('dataset_index') == data_index:
            tokens_used = detail['tokens_used']
            tokenizations.append(
                {
                    "candidate_description": detail.get('candidate_description'),
                    "tokens_used": tokens_used,
                    "trainness_score": calculator.get_score(tokens_used),
                    "renyi": renyi_score(tokens_used),
                    "score": detail.get('score')
                }
            )

    return tokenizations

def plot_tokenization_metrics(tokenizations_1: list, tokenizations_2: list,
                               label1: str = "Data 1", label2: str = "Data 2",
                               annotate: bool = False,
                               hover: bool = False):
    """
    Scatter plot to compare tokenization metrics for MT.
    x-axis: trainness_score, y-axis: Rényi score, color: BLEU score.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    all_scores = [item['score'] for item in tokenizations_1 + tokenizations_2 if item.get('score') is not None]
    vmin = min(all_scores) if all_scores else 0
    vmax = max(all_scores) if all_scores else 1

    scatter_collections = []
    all_items = []

    def plot_group(data, marker, label):
        if not data:
            return
        
        scores = [item.get('score') for item in data]
        sc = ax.scatter(
            [item['trainness_score'] for item in data],
            [item['renyi'] for item in data],
            c=scores,
            cmap='viridis',
            marker=marker,
            edgecolors='black',
            linewidths=0.6,
            s=80,
            alpha=0.85,
            label=label,
            vmin=vmin,
            vmax=vmax
        )
        scatter_collections.append(sc)
        all_items.extend(data)
        
        if annotate:
            for item in data:
                ax.text(item['trainness_score'], item['renyi'],
                        item.get('candidate_description', ''),
                        fontsize=7, alpha=0.7, ha='left', va='bottom')

    plot_group(tokenizations_1, 'o', label1)
    plot_group(tokenizations_2, 's', label2)

    ax.set_xlabel('Trainness Score', fontsize=12)
    ax.set_ylabel('Rényi Score', fontsize=12)
    ax.set_title('Trainness vs. Rényi Score (Color by BLEU)', fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(frameon=False, fontsize=9)

    if all_scores:
        cbar = fig.colorbar(scatter_collections[0], ax=ax)
        cbar.set_label('BLEU Score')

    if hover:
        if mplcursors is None:
            print("mplcursors not installed. Install with: pip install mplcursors")
        else:
            cursor = mplcursors.cursor(scatter_collections, hover=True)
            @cursor.connect("add")
            def on_add(sel):
                item_index = sel.target.index
                artist_collection = sel.artist
                start_index = 0
                for sc in scatter_collections:
                    if artist_collection == sc:
                        item = all_items[start_index + item_index]
                        break
                    start_index += len(sc.get_offsets())
                
                text = (
                    f"Desc: {item.get('candidate_description', 'N/A')}\n"
                    f"BLEU: {item.get('score', -1):.2f}\n"
                    f"Tokens: {' '.join(item.get('tokens_used', []))}"
                )
                sel.annotation.set_text(text)
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9, edgecolor="#333")

    plt.tight_layout()
    plt.show()

def main(args):
    raw_data_1 = read_json(args.input_1)["sample_translations"]
    raw_data_2 = read_json(args.input_2)["sample_translations"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if args.entropy_file:
        calculator = TokenEntropy(args.entropy_file, tokenizer)
    elif args.magikarp_file:
        calculator = TokenNorm(args.magikarp_file, tokenizer)
    else:
        print("Error: Either --entropy_file or --magikarp_file must be provided.")
        return

    tokenizations_1 = get_tokenizations_data(
        data_index=args.data_index, raw_data=raw_data_1, calculator=calculator)
    tokenizations_2 = get_tokenizations_data(
        data_index=args.data_index, raw_data=raw_data_2, calculator=calculator)

    plot_tokenization_metrics(
        tokenizations_1,
        tokenizations_2,
        label1=os.path.basename(args.input_1),
        label2=os.path.basename(args.input_2),
        annotate=args.annotate,
        hover=args.hover
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare tokenizations for a specific data sample in MT.")
    parser.add_argument("--input_1", required=True, help="Path to first JSON file.")
    parser.add_argument("--input_2", required=True, help="Path to second JSON file.")
    parser.add_argument("--data_index", type=int, default=121, help="Index of the data sample to analyze.")
    parser.add_argument("--annotate", action="store_true", help="Annotate points with candidate descriptions.")
    parser.add_argument('--magikarp_file', type=str, help="Path to Magikarp JSONL file.")
    parser.add_argument('--entropy_file', type=str, help="Path to token entropy JSON file.")
    parser.add_argument('--tokenizer_name', type=str, required=True, help="Tokenizer name or path.")
    parser.add_argument('--hover', action="store_true", help="Enable hover tooltips (requires mplcursors).")

    args = parser.parse_args()
    main(args)