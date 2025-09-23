import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from typing import Dict, List

def load_json_file(filepath: str) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_tokenization_results(baseline_file: str, alternative_file: str) -> Dict:
    baseline_data = load_json_file(baseline_file)
    alternative_data = load_json_file(alternative_file)
    
    baseline_samples = baseline_data['sample_translations']
    alternative_samples = alternative_data['sample_translations']
    
    comparison_results = []
    better_count = worse_count = same_count = 0
    
    min_samples = min(len(baseline_samples), len(alternative_samples))
    
    for i in range(min_samples):
        baseline_sample = baseline_samples[i]
        alternative_sample = alternative_samples[i]
        
        baseline_score = baseline_sample['score']
        alternative_score = alternative_sample['score']
        score_difference = alternative_score - baseline_score
        
        if score_difference > 0:
            improvement = "Better"
            better_count += 1
        elif score_difference < 0:
            improvement = "Worse"
            worse_count += 1
        else:
            improvement = "Same"
            same_count += 1
        
        comparison_results.append({
            'dataset_index': baseline_sample['dataset_index'],
            'baseline_score': baseline_score,
            'alternative_score': alternative_score,
            'score_difference': score_difference,
            'improvement': improvement,
            'baseline_text': baseline_sample['predicted_text'],
            'alternative_text': alternative_sample['predicted_text'],
            'baseline_problem': baseline_sample.get('problem', False),
            'alternative_problem': alternative_sample.get('problem', False)
        })
    
    total_comparisons = len(comparison_results)
    better_percentage = (better_count / total_comparisons) * 100
    worse_percentage = (worse_count / total_comparisons) * 100
    same_percentage = (same_count / total_comparisons) * 100
    
    score_differences = [result['score_difference'] for result in comparison_results]
    avg_score_difference = sum(score_differences) / len(score_differences)
    
    baseline_overall = baseline_data['overall_score']['score']
    alternative_overall = alternative_data['overall_score']['score']
    overall_difference = alternative_overall - baseline_overall
    
    return {
        'summary': {
            'total_comparisons': total_comparisons,
            'better_count': better_count,
            'worse_count': worse_count,
            'same_count': same_count,
            'better_percentage': better_percentage,
            'worse_percentage': worse_percentage,
            'same_percentage': same_percentage,
            'avg_score_difference': avg_score_difference,
            'baseline_overall_score': baseline_overall,
            'alternative_overall_score': alternative_overall,
            'overall_score_difference': overall_difference
        },
        'detailed_results': comparison_results,
        'baseline_model': baseline_data['model_name'],
        'alternative_model': alternative_data['model_name']
    }

def generate_analysis_report(comparison_data: Dict) -> None:
    summary = comparison_data['summary']
    
    print("=" * 80)
    print("TOKENIZATION COMPARISON ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\nModels Compared:")
    print(f"  Baseline: {comparison_data['baseline_model']}")
    print(f"  Alternative: {comparison_data['alternative_model']}")
    
    print(f"\nOverall Score Comparison:")
    print(f"  Baseline Overall Score: {summary['baseline_overall_score']:.4f}")
    print(f"  Alternative Overall Score: {summary['alternative_overall_score']:.4f}")
    print(f"  Overall Difference: {summary['overall_score_difference']:.4f}")
    
    print(f"\nSample-by-Sample Comparison ({summary['total_comparisons']} samples):")
    print(f"  Better Results: {summary['better_count']} ({summary['better_percentage']:.2f}%)")
    print(f"  Worse Results: {summary['worse_count']} ({summary['worse_percentage']:.2f}%)")
    print(f"  Same Results: {summary['same_count']} ({summary['same_percentage']:.2f}%)")
    
    print(f"\nAverage Score Difference: {summary['avg_score_difference']:.4f}")
    
    detailed = comparison_data['detailed_results']
    most_improved = max(detailed, key=lambda x: x['score_difference'])
    most_degraded = min(detailed, key=lambda x: x['score_difference'])
    
    print(f"\nMost Improved Sample (Index {most_improved['dataset_index']}):")
    print(f"  Score Improvement: +{most_improved['score_difference']:.4f}")
    print(f"  Baseline: {most_improved['baseline_text'][:100]}...")
    print(f"  Alternative: {most_improved['alternative_text'][:100]}...")
    
    print(f"\nMost Degraded Sample (Index {most_degraded['dataset_index']}):")
    print(f"  Score Degradation: {most_degraded['score_difference']:.4f}")
    print(f"  Baseline: {most_degraded['baseline_text'][:100]}...")
    print(f"  Alternative: {most_degraded['alternative_text'][:100]}...")

def create_visualizations(comparison_data: Dict, save_plots: bool = False, output_dir: str = '.') -> None:
    detailed = comparison_data['detailed_results']
    summary = comparison_data['summary']
    df = pd.DataFrame(detailed)
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tokenization Comparison Analysis', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    labels = ['Better', 'Worse', 'Same']
    sizes = [summary['better_count'], summary['worse_count'], summary['same_count']]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Distribution of Results')
    
    ax2 = axes[0, 1]
    ax2.hist(df['score_difference'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax2.set_xlabel('Score Difference (Alternative - Baseline)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Score Differences')
    ax2.legend()
    
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['baseline_score'], df['alternative_score'], 
                         alpha=0.6, c=df['score_difference'], cmap='RdYlGn')
    ax3.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Correlation')
    ax3.set_xlabel('Baseline Score')
    ax3.set_ylabel('Alternative Score')
    ax3.set_title('Baseline vs Alternative Scores')
    ax3.legend()
    plt.colorbar(scatter, ax=ax3, label='Score Difference')
    
    ax4 = axes[1, 1]
    improvement_data = [
        df[df['improvement'] == 'Better']['score_difference'],
        df[df['improvement'] == 'Worse']['score_difference'],
        df[df['improvement'] == 'Same']['score_difference']
    ]
    ax4.boxplot(improvement_data, labels=['Better', 'Worse', 'Same'])
    ax4.set_ylabel('Score Difference')
    ax4.set_title('Score Differences by Category')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = os.path.join(output_dir, 'tokenization_comparison_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_path}")
    
    plt.show()

def save_detailed_results_to_csv(comparison_data: Dict, output_file: str) -> None:
    df = pd.DataFrame(comparison_data['detailed_results'])
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")

def analyze_problem_samples(detailed_results: List[Dict]) -> Dict:
    return {
        'baseline_problems': sum(1 for r in detailed_results if r['baseline_problem']),
        'alternative_problems': sum(1 for r in detailed_results if r['alternative_problem']),
        'problems_fixed': sum(1 for r in detailed_results if r['baseline_problem'] and not r['alternative_problem']),
        'problems_introduced': sum(1 for r in detailed_results if not r['baseline_problem'] and r['alternative_problem']),
        'problems_persisted': sum(1 for r in detailed_results if r['baseline_problem'] and r['alternative_problem'])
    }

def print_problem_analysis(problem_stats: Dict) -> None:
    print("\n" + "=" * 50)
    print("PROBLEM SAMPLES ANALYSIS")
    print("=" * 50)
    print(f"Baseline Problems: {problem_stats['baseline_problems']}")
    print(f"Alternative Problems: {problem_stats['alternative_problems']}")
    print(f"Problems Fixed: {problem_stats['problems_fixed']}")
    print(f"Problems Introduced: {problem_stats['problems_introduced']}")
    print(f"Problems Persisted: {problem_stats['problems_persisted']}")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Compare baseline and alternative tokenization results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tokenization.py -b baseline.json -a alternative.json
  python tokenization.py -b baseline.json -a alternative.json --no-plots
  python tokenization.py -b baseline.json -a alternative.json -o results --save-plots
  python tokenization.py -b baseline.json -a alternative.json --csv-only
        """
    )
    
    parser.add_argument('-b', '--baseline', type=str, required=True,
                       help='Path to baseline tokenization results JSON file')
    parser.add_argument('-a', '--alternative', type=str, required=True,
                       help='Path to alternative tokenization results JSON file')
    parser.add_argument('-o', '--output', type=str, default='tokenization_comparison',
                       help='Output filename prefix (default: tokenization_comparison)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for results (default: current directory)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots/visualizations')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to file instead of just displaying')
    parser.add_argument('--no-csv', action='store_true',
                       help='Skip saving CSV file')
    parser.add_argument('--csv-only', action='store_true',
                       help='Only generate CSV file, skip report and plots')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output (only save files)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if not os.path.exists(args.baseline):
        print(f"Error: Baseline file not found: {args.baseline}")
        return 1
    
    if not os.path.exists(args.alternative):
        print(f"Error: Alternative file not found: {args.alternative}")
        return 1
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        if not args.quiet:
            print(f"Created output directory: {args.output_dir}")
    
    try:
        if not args.quiet:
            print("Loading and comparing tokenization results...")
        
        comparison_data = compare_tokenization_results(args.baseline, args.alternative)
        
        if args.csv_only:
            csv_path = os.path.join(args.output_dir, f"{args.output}.csv")
            save_detailed_results_to_csv(comparison_data, csv_path)
        else:
            if not args.quiet:
                generate_analysis_report(comparison_data)
                
                detailed = comparison_data['detailed_results']
                problem_analysis = analyze_problem_samples(detailed)
                print_problem_analysis(problem_analysis)
            
            if not args.no_csv:
                csv_path = os.path.join(args.output_dir, f"{args.output}.csv")
                save_detailed_results_to_csv(comparison_data, csv_path)
            
            if not args.no_plots:
                create_visualizations(comparison_data, args.save_plots, args.output_dir)
        
        if not args.quiet:
            print(f"\nAnalysis completed successfully!")
            
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return 1
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1

if __name__ == "__main__":
    exit(main())