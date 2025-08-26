import os
import argparse
import re
import json

import matplotlib.pyplot as plt
import numpy as np

def read_json(file_path: str) -> dict:
    """Reads a JSON file and returns its content as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def extract_accuracy(text: str) -> [float, float]:
    """Extract accuracy and number of correct from string 'accuracy (number of correct / total question)"""
    match = re.search(r"(\d+\.\d+)\s\((\d+)/\d+\)", text)
    if match:
        accuracy = float(match.group(1))
        num_correct = int(match.group(2))
        return accuracy, num_correct
    
    return None, None

def get_subject_data(subject_raw: dict) -> dict:
    """Read a JSON file about performance in each subject and return clean dict"""
    clean_data = {}
    for subject, details in subject_raw.items():
        accuracy, num_correct = extract_accuracy(details)
        clean_data[subject] = {
            "accuracy": accuracy,
            "num_correct": num_correct
        }
    return clean_data

def get_overall_data(detail: str) -> dict:
    """Read a formatted text and extract accuracy and total number of correct"""
    accuracy, num_correct = extract_accuracy(detail)
    return {
        "accuracy": accuracy,
        "num_correct": num_correct
    }

def plot_subject_comparison(data1: dict, data2: dict, label1: str = 'Data 1', label2: str = 'Data 2', sort_by: str = 'subject', subjects_per_plot: int = 30):
    """
    Generates and displays side-by-side horizontal bar charts, splitting subjects 
    across multiple plots to ensure readability.
    """
    all_subjects = set(data1.keys()) | set(data2.keys())
    plot_data = []
    for subject in all_subjects:
        acc1 = data1.get(subject, {}).get('accuracy', 0)
        acc2 = data2.get(subject, {}).get('accuracy', 0)
        plot_data.append({
            'subject': subject,
            'acc1': acc1,
            'acc2': acc2,
            'diff': abs(acc1 - acc2)
        })

    
    reverse_sort = True
    if sort_by == 'subject':
        sort_key = 'subject'
        reverse_sort = False
    elif sort_by == 'data1':
        sort_key = 'acc1'
    elif sort_by == 'data2':
        sort_key = 'acc2'
    elif sort_by == 'diff':
        sort_key = 'diff'
    else:
        raise ValueError("sort_by must be one of: 'subject', 'data1', 'data2', 'diff'")
    
    plot_data.sort(key=lambda item: item[sort_key], reverse=reverse_sort)

    
    total_subjects = len(plot_data)
    num_plots = (total_subjects + subjects_per_plot - 1) // subjects_per_plot

    for i in range(num_plots):
        start_index = i * subjects_per_plot
        end_index = start_index + subjects_per_plot
        chunk = plot_data[start_index:end_index]

        subjects = [item['subject'] for item in chunk]
        accuracies1 = [item['acc1'] for item in chunk]
        accuracies2 = [item['acc2'] for item in chunk]

        
        fig_height = len(subjects) * 0.6 
        y_pos = np.arange(len(subjects))
        bar_height = 0.35

        fig, ax = plt.subplots(figsize=(12, fig_height))
        
        bars1 = ax.barh(y_pos - bar_height/2, accuracies1, bar_height, label=label1, color='steelblue')
        bars2 = ax.barh(y_pos + bar_height/2, accuracies2, bar_height, label=label2, color='lightcoral')

        ax.set_xlabel('Accuracy', fontsize=12)
        title = f'Comparison of Accuracy by Subject (Part {i+1} of {num_plots})'
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subjects, fontsize=10)
        ax.invert_yaxis()
        ax.legend()
        
        ax.bar_label(bars1, padding=3, fmt='%.3f', fontsize=9)
        ax.bar_label(bars2, padding=3, fmt='%.3f', fontsize=9)
        
        ax.set_xlim(0, 1.1)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

def plot_accuracy_change(
    data1: dict,
    data2: dict,
    label1: str = 'Data 1',
    label2: str = 'Data 2',
    sort_by: str = 'delta',
    subjects_per_plot: int = 30,
    zero_center: bool = True,
    title: str | None = None
):
    """
    Plot per-subject accuracy change (data2 - data1) as a horizontal diverging bar chart.
    """
    subjects = sorted(set(data1.keys()) | set(data2.keys()))
    rows = []
    for s in subjects:
        acc1 = (data1.get(s, {}) or {}).get('accuracy') or 0.0
        acc2 = (data2.get(s, {}) or {}).get('accuracy') or 0.0
        delta = acc2 - acc1
        rows.append({'subject': s, 'acc1': acc1, 'acc2': acc2, 'delta': delta, 'abs': abs(delta)})

    if sort_by == 'delta':
        rows.sort(key=lambda r: r['delta'], reverse=True)
    elif sort_by == 'abs':
        rows.sort(key=lambda r: r['abs'], reverse=True)
    elif sort_by == 'subject':
        rows.sort(key=lambda r: r['subject'].lower())
    else:
        raise ValueError("sort_by must be one of {'delta','abs','subject'}")

    total = len(rows)
    num_plots = (total + subjects_per_plot - 1) // subjects_per_plot

    for i in range(num_plots):
        chunk = rows[i*subjects_per_plot:(i+1)*subjects_per_plot]
        subj = [r['subject'] for r in chunk]
        delta_vals = [r['delta'] for r in chunk]

        y = np.arange(len(chunk))
        colors = ['#2ca02c' if d > 0 else '#d62728' if d < 0 else '#7f7f7f' for d in delta_vals]

        fig_height = max(2.5, 0.5 * len(chunk))
        fig, ax = plt.subplots(figsize=(12, fig_height))

        bars = ax.barh(y, delta_vals, color=colors)

        ax.set_yticks(y)
        ax.set_yticklabels(subj, fontsize=9)
        ax.invert_yaxis()

        for bar, d in zip(bars, delta_vals):
            ax.text(
                bar.get_width() + (0.005 if d >= 0 else -0.005),
                bar.get_y() + bar.get_height()/2,
                f"{d:+.3f}",
                va='center',
                ha='left' if d >= 0 else 'right',
                fontsize=8
            )

        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel(f'Accuracy change ({label2} - {label1})')
        auto_title = f'Per-subject accuracy change ({i+1} / {num_plots})'
        ax.set_title(title or auto_title, fontweight='bold')

        max_abs = max(0.01, max(abs(d) for d in delta_vals))
        if zero_center:
            ax.set_xlim(-max_abs * 1.05, max_abs * 1.05)
        else:
            left = min(0, min(delta_vals))*1.05
            right = max(0, max(delta_vals))*1.05
            ax.set_xlim(left, right)

        ax.grid(axis='x', linestyle='--', alpha=0.4)

        plt.tight_layout()
        plt.show()

def main(args):
    raw_data_1 = read_json(args.input_1)
    raw_data_2 = read_json(args.input_2)

    subject_data_1 = get_subject_data(raw_data_1["subject_accuracies"])
    subject_data_2 = get_subject_data(raw_data_2["subject_accuracies"])

    overall_data_1 = get_overall_data(raw_data_1["overall_accuracy"])
    overall_data_2 = get_overall_data(raw_data_2["overall_accuracy"])

    plot_subject_comparison(
        subject_data_1, subject_data_2,
        label1=os.path.basename(args.input_1), label2=os.path.basename(args.input_2),
        sort_by='subject',      
        subjects_per_plot=20  
    )

    plot_accuracy_change(
        subject_data_1, subject_data_2,
        label1=os.path.basename(args.input_1), label2=os.path.basename(args.input_2),
        sort_by='delta',
        subjects_per_plot=30
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and compare performance data from two JSON files.")
    parser.add_argument("--input_1", type=str, required=True, help="Path to the first input JSON file.")
    parser.add_argument("--input_2", type=str, required=True, help="Path to the second input JSON file.")
    args = parser.parse_args()
    main(args)
