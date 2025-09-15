import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import re
from typing import Dict, Tuple, Optional


class BenchmarkComparator:
    def __init__(self, file1_path: str, file2_path: str):
        """
        Initialize the comparator with two JSON benchmark files.
        
        Args:
            file1_path: Path to first benchmark JSON file
            file2_path: Path to second benchmark JSON file
        """
        self.file1_path = file1_path
        self.file2_path = file2_path
        self.data1 = self._load_json(file1_path)
        self.data2 = self._load_json(file2_path)
        
    def _load_json(self, file_path: str) -> dict:
        """Load JSON data from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _parse_accuracy(self, accuracy_str: str) -> Optional[float]:
        """
        Parse accuracy string to extract float value.
        Handles formats like "0.8372 (36/43)" or "N/A (...)"
        """
        if accuracy_str.startswith("N/A"):
            return None
        
        # Extract the float value before the parentheses
        match = re.match(r'^(\d+\.\d+)', accuracy_str)
        if match:
            return float(match.group(1))
        return None
    
    def _get_accuracy_data(self, data: dict) -> Dict[str, float]:
        """Extract accuracy data from benchmark results."""
        accuracies = {}
        
        # Check for both possible key names
        category_key = None
        if "category_accuracies" in data:
            category_key = "category_accuracies"
        elif "subject_accuracies" in data:
            category_key = "subject_accuracies"
        
        if category_key:
            for category, accuracy_str in data[category_key].items():
                accuracy = self._parse_accuracy(accuracy_str)
                if accuracy is not None:
                    accuracies[category] = accuracy
        
        # Add overall accuracy if available
        if "overall_accuracy" in data:
            overall_acc = self._parse_accuracy(data["overall_accuracy"])
            if overall_acc is not None:
                accuracies["Overall"] = overall_acc
                
        return accuracies
    
    def create_comparison_dataframe(self, include_overall: bool = False) -> pd.DataFrame:
        """Create a DataFrame comparing the two benchmark results."""
        acc1 = self._get_accuracy_data(self.data1)
        acc2 = self._get_accuracy_data(self.data2)
        
        # Get common categories
        common_categories = set(acc1.keys()) & set(acc2.keys())
        
        # Exclude Overall category unless specifically requested
        if not include_overall and "Overall" in common_categories:
            common_categories.remove("Overall")
        
        comparison_data = []
        for category in common_categories:
            comparison_data.append({
                'Category': category,
                'Model1_Accuracy': acc1[category],
                'Model2_Accuracy': acc2[category],
                'Difference': acc2[category] - acc1[category],
                'Improvement_Pct': ((acc2[category] - acc1[category]) / acc1[category]) * 100
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Difference', ascending=False)
    
    def get_overall_comparison(self) -> Optional[Dict[str, float]]:
        """Get overall accuracy comparison separately."""
        acc1 = self._get_accuracy_data(self.data1)
        acc2 = self._get_accuracy_data(self.data2)
        
        if "Overall" in acc1 and "Overall" in acc2:
            return {
                'Model1_Overall': acc1["Overall"],
                'Model2_Overall': acc2["Overall"],
                'Overall_Difference': acc2["Overall"] - acc1["Overall"],
                'Overall_Improvement_Pct': ((acc2["Overall"] - acc1["Overall"]) / acc1["Overall"]) * 100
            }
        return None
    
    def plot_side_by_side_comparison(self, figsize: Tuple[int, int] = (15, 10)):
        """Create a side-by-side bar chart comparison (excluding Overall)."""
        df = self.create_comparison_dataframe(include_overall=False)
        
        # Get model names from data
        model1_name = self.data1.get("model_name", "Model 1")
        model2_name = self.data2.get("model_name", "Model 2")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df['Model1_Accuracy'], width, 
                      label=model1_name, alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, df['Model2_Accuracy'], width, 
                      label=model2_name, alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Categories')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Category Accuracy Comparison: {model1_name} vs {model2_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Category'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_overall_comparison(self, figsize: Tuple[int, int] = (8, 6)):
        """Create a separate chart for overall accuracy comparison."""
        overall_data = self.get_overall_comparison()
        
        if overall_data is None:
            print("No overall accuracy data available for comparison")
            return None
        
        model1_name = self.data1.get("model_name", "Model 1")
        model2_name = self.data2.get("model_name", "Model 2")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        models = [model1_name, model2_name]
        accuracies = [overall_data['Model1_Overall'], overall_data['Model2_Overall']]
        
        colors = ['skyblue', 'lightcoral']
        bars = ax.bar(models, accuracies, color=colors, alpha=0.8, width=0.6)
        
        ax.set_ylabel('Overall Accuracy')
        ax.set_title('Overall Accuracy Comparison')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add difference annotation
        diff = overall_data['Overall_Difference']
        improvement_pct = overall_data['Overall_Improvement_Pct']
        
        ax.text(0.5, max(accuracies) * 0.95, 
               f'Difference: {diff:+.4f} ({improvement_pct:+.2f}%)',
               ha='center', va='top', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_difference_chart(self, figsize: Tuple[int, int] = (12, 8)):
        """Create a chart showing the difference between models (excluding Overall)."""
        df = self.create_comparison_dataframe(include_overall=False)
        
        model1_name = self.data1.get("model_name", "Model 1")
        model2_name = self.data2.get("model_name", "Model 2")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color bars based on improvement (green) or decline (red)
        colors = ['green' if x > 0 else 'red' for x in df['Difference']]
        
        bars = ax.barh(df['Category'], df['Difference'], color=colors, alpha=0.7)
        
        ax.set_xlabel('Accuracy Difference')
        ax.set_ylabel('Categories')
        ax.set_title(f'Category Performance Difference: {model2_name} - {model1_name}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, diff) in enumerate(zip(bars, df['Difference'])):
            ax.text(bar.get_width() + (0.005 if diff > 0 else -0.005), 
                   bar.get_y() + bar.get_height()/2,
                   f'{diff:.3f}', ha='left' if diff > 0 else 'right', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_heatmap_comparison(self, figsize: Tuple[int, int] = (10, 8)):
        """Create a heatmap showing accuracy values for both models (excluding Overall)."""
        df = self.create_comparison_dataframe(include_overall=False)
        
        model1_name = self.data1.get("model_name", "Model 1")
        model2_name = self.data2.get("model_name", "Model 2")
        
        # Prepare data for heatmap
        heatmap_data = df[['Model1_Accuracy', 'Model2_Accuracy']].T
        heatmap_data.columns = df['Category']
        heatmap_data.index = [model1_name, model2_name]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0.5, ax=ax, cbar_kws={'label': 'Accuracy'})
        
        ax.set_title('Category Accuracy Heatmap Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self) -> str:
        """Generate a text summary of the comparison."""
        df = self.create_comparison_dataframe(include_overall=False)
        overall_data = self.get_overall_comparison()
        
        model1_name = self.data1.get("model_name", "Model 1")
        model2_name = self.data2.get("model_name", "Model 2")
        
        report = f"Benchmark Comparison Report\n"
        report += f"{'='*50}\n\n"
        report += f"Model 1: {model1_name}\n"
        report += f"Model 2: {model2_name}\n\n"
        
        # Overall accuracy comparison
        if overall_data:
            report += f"Overall Accuracy Comparison:\n"
            report += f"- {model1_name}: {overall_data['Model1_Overall']:.4f}\n"
            report += f"- {model2_name}: {overall_data['Model2_Overall']:.4f}\n"
            report += f"- Difference: {overall_data['Overall_Difference']:+.4f} ({overall_data['Overall_Improvement_Pct']:+.2f}%)\n\n"
        
        # Category statistics
        avg_diff = df['Difference'].mean()
        max_improvement = df['Difference'].max()
        max_decline = df['Difference'].min()
        
        report += f"Category Summary Statistics:\n"
        report += f"- Average difference: {avg_diff:.4f}\n"
        report += f"- Maximum improvement: {max_improvement:.4f}\n"
        report += f"- Maximum decline: {max_decline:.4f}\n"
        report += f"- Categories improved: {sum(df['Difference'] > 0)}\n"
        report += f"- Categories declined: {sum(df['Difference'] < 0)}\n"
        report += f"- Total categories compared: {len(df)}\n\n"
        
        # Top improvements and declines
        report += "Top 5 Category Improvements:\n"
        top_improvements = df.nlargest(5, 'Difference')
        for _, row in top_improvements.iterrows():
            report += f"- {row['Category']}: +{row['Difference']:.4f} ({row['Improvement_Pct']:.2f}%)\n"
        
        report += "\nTop 5 Category Declines:\n"
        top_declines = df.nsmallest(5, 'Difference')
        for _, row in top_declines.iterrows():
            report += f"- {row['Category']}: {row['Difference']:.4f} ({row['Improvement_Pct']:.2f}%)\n"
        
        return report
    
    def save_all_visualizations(self, output_dir: str = "comparison_results"):
        """Save all visualizations and report to files."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save category comparison chart
        fig1 = self.plot_side_by_side_comparison()
        fig1.savefig(f"{output_dir}/category_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Save overall comparison chart
        fig_overall = self.plot_overall_comparison()
        if fig_overall:
            fig_overall.savefig(f"{output_dir}/overall_comparison.png", dpi=300, bbox_inches='tight')
            plt.close(fig_overall)
        
        # Save difference chart
        fig2 = self.plot_difference_chart()
        fig2.savefig(f"{output_dir}/difference_chart.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # Save heatmap
        fig3 = self.plot_heatmap_comparison()
        fig3.savefig(f"{output_dir}/heatmap_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        # Save category DataFrame
        df_categories = self.create_comparison_dataframe(include_overall=False)
        df_categories.to_csv(f"{output_dir}/category_comparison_data.csv", index=False)
        
        # Save overall comparison data
        overall_data = self.get_overall_comparison()
        if overall_data:
            overall_df = pd.DataFrame([overall_data])
            overall_df.to_csv(f"{output_dir}/overall_comparison_data.csv", index=False)
        
        # Save summary report
        report = self.generate_summary_report()
        with open(f"{output_dir}/summary_report.txt", 'w') as f:
            f.write(report)
        
        print(f"All visualizations and data saved to '{output_dir}' directory")

# Example usage function
def compare_benchmarks(file1_path: str, file2_path: str, show_plots: bool = True):
    """
    Main function to compare two benchmark files.
    
    Args:
        file1_path: Path to first benchmark JSON file
        file2_path: Path to second benchmark JSON file
        show_plots: Whether to display plots immediately
    """
    comparator = BenchmarkComparator(file1_path, file2_path)
    
    if show_plots:
        # Display overall comparison first
        comparator.plot_overall_comparison()
        plt.show()
        
        # Display category plots
        comparator.plot_side_by_side_comparison()
        plt.show()
        
        comparator.plot_difference_chart()
        plt.show()
        
        comparator.plot_heatmap_comparison()
        plt.show()
    
    # Print summary report
    print(comparator.generate_summary_report())
    
    # Save everything
    comparator.save_all_visualizations()
    
    return comparator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument('--file1', type=str, help='Path to first benchmark JSON file')
    parser.add_argument('--file2', type=str, help='Path to second benchmark JSON file')

    args = parser.parse_args()
    file1 = args.file1
    file2 = args.file2

    comparator = compare_benchmarks(file1, file2)