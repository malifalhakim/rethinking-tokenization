import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import re
from typing import Dict, List
from collections import Counter
import numpy as np

def load_json_file(filepath: str) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_undertrained_tokens(tokens: List[str]) -> List[str]:
    """Extract tokens between 'Question:' and '--' which represent undertrained words."""
    try:
        question_start_idx = None
        for i, token in enumerate(tokens):
            if token == ":" and i > 0 and "Question" in tokens[i-1]:
                question_start_idx = i
                break
            elif token == "Question:" or token == "Question":
                question_start_idx = i
                break
        
        if question_start_idx is None:
            return []
        
        dash_idx = None
        for i, token in enumerate(tokens[question_start_idx:], question_start_idx):
            if token == "\u0120--" or token == "--":
                dash_idx = i
                break
        
        if dash_idx is None:
            return []
            
        return tokens[question_start_idx + 1:dash_idx]
    except (ValueError, IndexError):
        return []

def analyze_undertrained_tokenization(tokens: List[str]) -> Dict:
    """Analyze undertrained word tokenization patterns."""
    undertrained_tokens = extract_undertrained_tokens(tokens)
    
    filtered_tokens = [t for t in undertrained_tokens if t not in ["\u010a", "\u0120", ""]]
    
    return {
        'undertrained_token_count': len(filtered_tokens),
        'undertrained_tokens': filtered_tokens,
        'has_character_level_splits': any(len(t.strip('\u0120')) == 1 for t in filtered_tokens if t.strip('\u0120').isalpha()),
        'avg_token_length': np.mean([len(t.strip('\u0120')) for t in filtered_tokens]) if filtered_tokens else 0,
        'single_char_tokens': sum(1 for t in filtered_tokens if len(t.strip('\u0120')) == 1 and t.strip('\u0120').isalpha())
    }

def compare_tokenization_results(baseline_file: str, alternative_file: str) -> Dict:
    baseline_data = load_json_file(baseline_file)
    alternative_data = load_json_file(alternative_file)
    
    baseline_samples = baseline_data['per_candidate_results']
    alternative_samples = alternative_data['per_candidate_results']
    
    comparison_results = []
    better_count = worse_count = same_count = 0
    
    min_samples = min(len(baseline_samples), len(alternative_samples))
    
    for i in range(min_samples):
        baseline_sample = baseline_samples[i]
        alternative_sample = alternative_samples[i]
        
        baseline_score = 1.0 if baseline_sample['is_correct'] else 0.0
        alternative_score = 1.0 if alternative_sample['is_correct'] else 0.0
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
        
        baseline_tokenization = analyze_undertrained_tokenization(baseline_sample.get('tokens_used', []))
        alt_tokenization = analyze_undertrained_tokenization(alternative_sample.get('tokens_used', []))
        
        baseline_tokens_str = ' '.join(baseline_tokenization['undertrained_tokens'])
        alt_tokens_str = ' '.join(alt_tokenization['undertrained_tokens'])
        identical_tokenization = baseline_tokens_str == alt_tokens_str
        
        tokenization_similarity = calculate_tokenization_similarity(
            baseline_tokenization['undertrained_tokens'],
            alt_tokenization['undertrained_tokens']
        )
        
        comparison_results.append({
            'dataset_index': baseline_sample.get('question_index', i),
            'subject': baseline_sample.get('subject', 'unknown'),
            'baseline_score': baseline_score,
            'alternative_score': alternative_score,
            'score_difference': score_difference,
            'improvement': improvement,
            'baseline_correct': baseline_sample['is_correct'],
            'alternative_correct': alternative_sample['is_correct'],
            'baseline_predicted': baseline_sample.get('predicted_char', ''),
            'alternative_predicted': alternative_sample.get('predicted_char', ''),
            'correct_answer': baseline_sample.get('correct_char', ''),
            'undertrained_token_count': alt_tokenization['undertrained_token_count'],
            'undertrained_tokens': alt_tokenization['undertrained_tokens'],
            'has_character_level_splits': alt_tokenization['has_character_level_splits'],
            'avg_token_length': alt_tokenization['avg_token_length'],
            'single_char_tokens': alt_tokenization['single_char_tokens'],
            # Baseline tokenization info
            'baseline_undertrained_token_count': baseline_tokenization['undertrained_token_count'],
            'baseline_undertrained_tokens': baseline_tokenization['undertrained_tokens'],
            'baseline_has_character_level_splits': baseline_tokenization['has_character_level_splits'],
            'baseline_avg_token_length': baseline_tokenization['avg_token_length'],
            'baseline_single_char_tokens': baseline_tokenization['single_char_tokens'],
            # Tokenization comparison
            'identical_tokenization': identical_tokenization,
            'tokenization_similarity': tokenization_similarity,
            'token_count_difference': alt_tokenization['undertrained_token_count'] - baseline_tokenization['undertrained_token_count']
        })
    
    total_comparisons = len(comparison_results)
    better_percentage = (better_count / total_comparisons) * 100
    worse_percentage = (worse_count / total_comparisons) * 100
    same_percentage = (same_count / total_comparisons) * 100
    
    score_differences = [result['score_difference'] for result in comparison_results]
    avg_score_difference = sum(score_differences) / len(score_differences)
    
    baseline_accuracy = float(baseline_data['overall_accuracy'].split()[0])
    alternative_accuracy = float(alternative_data['overall_accuracy'].split()[0])
    overall_difference = alternative_accuracy - baseline_accuracy
    
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
            'baseline_overall_accuracy': baseline_accuracy,
            'alternative_overall_accuracy': alternative_accuracy,
            'overall_accuracy_difference': overall_difference
        },
        'detailed_results': comparison_results,
        'baseline_model': baseline_data.get('model_name', 'baseline'),
        'alternative_model': alternative_data.get('model_name', 'alternative')
    }

def calculate_tokenization_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Calculate similarity between two tokenizations using Jaccard similarity."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def analyze_same_score_tokenization(comparison_data: Dict) -> Dict:
    """Analyze tokenization patterns in cases where scores are the same."""
    detailed = comparison_data['detailed_results']
    same_cases = [r for r in detailed if r['improvement'] == 'Same']
    
    if not same_cases:
        return {"error": "No same score cases found for analysis"}
    
    identical_tokenization_count = sum(1 for r in same_cases if r['identical_tokenization'])
    similarity_scores = [r['tokenization_similarity'] for r in same_cases]
    token_count_diffs = [r['token_count_difference'] for r in same_cases]
    different_tokenization_cases = [r for r in same_cases if not r['identical_tokenization']]
    
    same_token_count_different_tokens = [
        r for r in same_cases 
        if r['token_count_difference'] == 0 and not r['identical_tokenization']
    ]
    
    different_token_count_same_score = [
        r for r in same_cases 
        if r['token_count_difference'] != 0
    ]
    
    baseline_char_splits = sum(1 for r in same_cases if r['baseline_has_character_level_splits'])
    alt_char_splits = sum(1 for r in same_cases if r['has_character_level_splits'])
    
    return {
        'total_same_cases': len(same_cases),
        'identical_tokenization_count': identical_tokenization_count,
        'identical_tokenization_rate': identical_tokenization_count / len(same_cases) * 100,
        'different_tokenization_count': len(different_tokenization_cases),
        'different_tokenization_rate': len(different_tokenization_cases) / len(same_cases) * 100,
        'similarity_stats': {
            'avg_similarity': np.mean(similarity_scores),
            'min_similarity': np.min(similarity_scores),
            'max_similarity': np.max(similarity_scores),
            'std_similarity': np.std(similarity_scores)
        },
        'token_count_difference_stats': {
            'avg_diff': np.mean(token_count_diffs),
            'min_diff': np.min(token_count_diffs),
            'max_diff': np.max(token_count_diffs),
            'std_diff': np.std(token_count_diffs),
            'zero_diff_count': sum(1 for d in token_count_diffs if d == 0)
        },
        'pattern_analysis': {
            'same_count_different_tokens': len(same_token_count_different_tokens),
            'different_count_same_score': len(different_token_count_same_score)
        },
        'character_splits_comparison': {
            'baseline_char_splits': baseline_char_splits,
            'alternative_char_splits': alt_char_splits,
            'baseline_char_split_rate': baseline_char_splits / len(same_cases) * 100,
            'alternative_char_split_rate': alt_char_splits / len(same_cases) * 100
        },
        'examples': {
            'identical_examples': [r for r in same_cases if r['identical_tokenization']][:3],
            'different_but_same_score_examples': different_tokenization_cases[:3],
            'same_count_different_tokens_examples': same_token_count_different_tokens[:3]
        }
    }

def print_same_score_analysis(same_score_stats: Dict) -> None:
    """Print detailed analysis of same score cases."""
    print("\n" + "=" * 80)
    print("SAME SCORE TOKENIZATION ANALYSIS")
    print("=" * 80)
    
    if 'error' in same_score_stats:
        print(f"Error: {same_score_stats['error']}")
        return
    
    total_same = same_score_stats['total_same_cases']
    identical_count = same_score_stats['identical_tokenization_count']
    different_count = same_score_stats['different_tokenization_count']
    
    print(f"\nOverall Same Score Cases: {total_same}")
    print(f"  Identical Tokenization: {identical_count} ({same_score_stats['identical_tokenization_rate']:.1f}%)")
    print(f"  Different Tokenization: {different_count} ({same_score_stats['different_tokenization_rate']:.1f}%)")
    
    similarity_stats = same_score_stats['similarity_stats']
    print(f"\nTokenization Similarity Analysis:")
    print(f"  Average Similarity: {similarity_stats['avg_similarity']:.3f}")
    print(f"  Min Similarity: {similarity_stats['min_similarity']:.3f}")
    print(f"  Max Similarity: {similarity_stats['max_similarity']:.3f}")
    print(f"  Std Deviation: {similarity_stats['std_similarity']:.3f}")
    
    token_diff_stats = same_score_stats['token_count_difference_stats']
    print(f"\nToken Count Difference Analysis:")
    print(f"  Average Difference: {token_diff_stats['avg_diff']:.2f}")
    print(f"  Min Difference: {token_diff_stats['min_diff']}")
    print(f"  Max Difference: {token_diff_stats['max_diff']}")
    print(f"  Cases with Same Token Count: {token_diff_stats['zero_diff_count']}")
    
    pattern_stats = same_score_stats['pattern_analysis']
    print(f"\nPattern Analysis:")
    print(f"  Same Token Count, Different Tokens: {pattern_stats['same_count_different_tokens']}")
    print(f"  Different Token Count, Same Score: {pattern_stats['different_count_same_score']}")
    
    char_stats = same_score_stats['character_splits_comparison']
    print(f"\nCharacter-Level Splits Comparison:")
    print(f"  Baseline: {char_stats['baseline_char_splits']}/{total_same} ({char_stats['baseline_char_split_rate']:.1f}%)")
    print(f"  Alternative: {char_stats['alternative_char_splits']}/{total_same} ({char_stats['alternative_char_split_rate']:.1f}%)")
    
    examples = same_score_stats['examples']
    
    if examples['identical_examples']:
        print(f"\nExamples of Identical Tokenization (Same Score):")
        for i, example in enumerate(examples['identical_examples'], 1):
            print(f"  {i}. Subject: {example['subject']}, Question {example['dataset_index']}: Correct={example['baseline_correct']}")
            print(f"     Tokens: {' '.join(example['undertrained_tokens'])}")
            print(f"     Answer: {example['baseline_predicted']} (Correct: {example['correct_answer']})")
    
    if examples['different_but_same_score_examples']:
        print(f"\nExamples of Different Tokenization (Same Score):")
        for i, example in enumerate(examples['different_but_same_score_examples'], 1):
            print(f"  {i}. Subject: {example['subject']}, Question {example['dataset_index']}: Correct={example['baseline_correct']}")
            print(f"     Baseline Tokens ({example['baseline_undertrained_token_count']}): {' '.join(example['baseline_undertrained_tokens'])}")
            print(f"     Alternative Tokens ({example['undertrained_token_count']}): {' '.join(example['undertrained_tokens'])}")
            print(f"     Similarity: {example['tokenization_similarity']:.3f}")
            print(f"     Answer: {example['alternative_predicted']} (Correct: {example['correct_answer']})")

def analyze_undertrained_impact(comparison_data: Dict) -> Dict:
    """Analyze the impact of undertrained tokenization on performance."""
    detailed = comparison_data['detailed_results']
    
    worse_cases = [r for r in detailed if r['improvement'] == 'Worse']
    better_cases = [r for r in detailed if r['improvement'] == 'Better']
    same_cases = [r for r in detailed if r['improvement'] == 'Same']
    
    if not worse_cases:
        return {"error": "No worse cases found for analysis"}
    
    worse_token_counts = [r['undertrained_token_count'] for r in worse_cases]
    better_token_counts = [r['undertrained_token_count'] for r in better_cases]
    same_token_counts = [r['undertrained_token_count'] for r in same_cases]
    
    worse_char_splits = sum(1 for r in worse_cases if r['has_character_level_splits'])
    better_char_splits = sum(1 for r in better_cases if r['has_character_level_splits'])
    same_char_splits = sum(1 for r in same_cases if r['has_character_level_splits'])
    
    worse_single_chars = [r['single_char_tokens'] for r in worse_cases]
    better_single_chars = [r['single_char_tokens'] for r in better_cases]
    same_single_chars = [r['single_char_tokens'] for r in same_cases]
    
    worse_avg_lengths = [r['avg_token_length'] for r in worse_cases if r['avg_token_length'] > 0]
    better_avg_lengths = [r['avg_token_length'] for r in better_cases if r['avg_token_length'] > 0]
    same_avg_lengths = [r['avg_token_length'] for r in same_cases if r['avg_token_length'] > 0]
    
    high_fragmentation_worse = [r for r in worse_cases if r['undertrained_token_count'] > 5]
    high_fragmentation_better = [r for r in better_cases if r['undertrained_token_count'] > 5]
    high_fragmentation_same = [r for r in same_cases if r['undertrained_token_count'] > 5]
    
    return {
        'worse_cases_count': len(worse_cases),
        'better_cases_count': len(better_cases),
        'same_cases_count': len(same_cases),
        'token_count_stats': {
            'worse_avg': np.mean(worse_token_counts) if worse_token_counts else 0,
            'worse_std': np.std(worse_token_counts) if worse_token_counts else 0,
            'worse_median': np.median(worse_token_counts) if worse_token_counts else 0,
            'better_avg': np.mean(better_token_counts) if better_token_counts else 0,
            'better_std': np.std(better_token_counts) if better_token_counts else 0,
            'better_median': np.median(better_token_counts) if better_token_counts else 0,
            'same_avg': np.mean(same_token_counts) if same_token_counts else 0,
            'same_std': np.std(same_token_counts) if same_token_counts else 0,
            'same_median': np.median(same_token_counts) if same_token_counts else 0
        },
        'character_splits': {
            'worse_cases_with_char_splits': worse_char_splits,
            'worse_cases_char_split_rate': worse_char_splits / len(worse_cases) * 100,
            'better_cases_with_char_splits': better_char_splits,
            'better_cases_char_split_rate': better_char_splits / len(better_cases) * 100 if better_cases else 0,
            'same_cases_with_char_splits': same_char_splits,
            'same_cases_char_split_rate': same_char_splits / len(same_cases) * 100 if same_cases else 0
        },
        'single_char_tokens': {
            'worse_avg': np.mean(worse_single_chars) if worse_single_chars else 0,
            'better_avg': np.mean(better_single_chars) if better_single_chars else 0,
            'same_avg': np.mean(same_single_chars) if same_single_chars else 0
        },
        'avg_token_length': {
            'worse_avg': np.mean(worse_avg_lengths) if worse_avg_lengths else 0,
            'better_avg': np.mean(better_avg_lengths) if better_avg_lengths else 0,
            'same_avg': np.mean(same_avg_lengths) if same_avg_lengths else 0
        },
        'high_fragmentation': {
            'worse_cases_count': len(high_fragmentation_worse),
            'better_cases_count': len(high_fragmentation_better),
            'same_cases_count': len(high_fragmentation_same),
            'worse_examples': high_fragmentation_worse[:3],
            'better_examples': high_fragmentation_better[:3],
            'same_examples': high_fragmentation_same[:3]
        }
    }

def analyze_subject_performance(comparison_data: Dict) -> Dict:
    """Analyze performance differences by subject."""
    detailed = comparison_data['detailed_results']
    
    # Group by subject
    subject_stats = {}
    for result in detailed:
        subject = result['subject']
        if subject not in subject_stats:
            subject_stats[subject] = {
                'total': 0,
                'better': 0,
                'worse': 0,
                'same': 0,
                'token_counts': [],
                'token_differences': []
            }
        
        subject_stats[subject]['total'] += 1
        subject_stats[subject][result['improvement'].lower()] += 1
        subject_stats[subject]['token_counts'].append(result['undertrained_token_count'])
        subject_stats[subject]['token_differences'].append(result['token_count_difference'])
    
    # Calculate percentages and averages
    for subject in subject_stats:
        stats = subject_stats[subject]
        total = stats['total']
        stats['better_pct'] = (stats['better'] / total) * 100
        stats['worse_pct'] = (stats['worse'] / total) * 100
        stats['same_pct'] = (stats['same'] / total) * 100
        stats['avg_tokens'] = np.mean(stats['token_counts'])
        stats['avg_token_diff'] = np.mean(stats['token_differences'])
    
    return subject_stats

def generate_analysis_report(comparison_data: Dict) -> None:
    summary = comparison_data['summary']
    
    print("=" * 80)
    print("MMLU TOKENIZATION COMPARISON ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\nModels Compared:")
    print(f"  Baseline: {comparison_data['baseline_model']}")
    print(f"  Alternative: {comparison_data['alternative_model']}")
    
    print(f"\nOverall Accuracy Comparison:")
    print(f"  Baseline Overall Accuracy: {summary['baseline_overall_accuracy']:.4f}")
    print(f"  Alternative Overall Accuracy: {summary['alternative_overall_accuracy']:.4f}")
    print(f"  Overall Difference: {summary['overall_accuracy_difference']:.4f}")
    
    print(f"\nSample-by-Sample Comparison ({summary['total_comparisons']} questions):")
    print(f"  Better Results: {summary['better_count']} ({summary['better_percentage']:.2f}%)")
    print(f"  Worse Results: {summary['worse_count']} ({summary['worse_percentage']:.2f}%)")
    print(f"  Same Results: {summary['same_count']} ({summary['same_percentage']:.2f}%)")
    
    print(f"\nAverage Score Difference: {summary['avg_score_difference']:.4f}")
    
    # Subject analysis
    subject_stats = analyze_subject_performance(comparison_data)
    print(f"\nSubject Performance Analysis:")
    print(f"{'Subject':<25} {'Total':<6} {'Better':<8} {'Worse':<7} {'Same':<6} {'Avg Tokens':<11}")
    print("-" * 70)
    
    for subject, stats in sorted(subject_stats.items()):
        print(f"{subject:<25} {stats['total']:<6} {stats['better']:<3}({stats['better_pct']:.1f}%) "
              f"{stats['worse']:<3}({stats['worse_pct']:.1f}%) {stats['same']:<3}({stats['same_pct']:.1f}%) "
              f"{stats['avg_tokens']:<11.1f}")

def print_undertrained_analysis(undertrained_stats: Dict) -> None:
    """Print detailed analysis of undertrained tokenization impact."""
    print("\n" + "=" * 80)
    print("UNDERTRAINED TOKENIZATION IMPACT ANALYSIS")
    print("=" * 80)
    
    if 'error' in undertrained_stats:
        print(f"Error: {undertrained_stats['error']}")
        return
    
    token_stats = undertrained_stats['token_count_stats']
    char_stats = undertrained_stats['character_splits']
    single_char_stats = undertrained_stats['single_char_tokens']
    avg_len_stats = undertrained_stats['avg_token_length']
    frag_stats = undertrained_stats['high_fragmentation']
    
    print(f"\nToken Count Analysis:")
    print(f"  Worse Cases ({undertrained_stats['worse_cases_count']}) - Avg: {token_stats['worse_avg']:.2f}, Median: {token_stats['worse_median']:.2f}, Std: {token_stats['worse_std']:.2f}")
    print(f"  Better Cases ({undertrained_stats['better_cases_count']}) - Avg: {token_stats['better_avg']:.2f}, Median: {token_stats['better_median']:.2f}, Std: {token_stats['better_std']:.2f}")
    print(f"  Same Cases ({undertrained_stats['same_cases_count']}) - Avg: {token_stats['same_avg']:.2f}, Median: {token_stats['same_median']:.2f}, Std: {token_stats['same_std']:.2f}")
    
    worse_vs_better_diff = token_stats['worse_avg'] - token_stats['better_avg']
    worse_vs_same_diff = token_stats['worse_avg'] - token_stats['same_avg']
    print(f"  → Worse vs Better: {worse_vs_better_diff:+.2f} tokens")
    print(f"  → Worse vs Same: {worse_vs_same_diff:+.2f} tokens")
    
    print(f"\nCharacter-Level Splits:")
    print(f"  Worse Cases: {char_stats['worse_cases_with_char_splits']}/{undertrained_stats['worse_cases_count']} ({char_stats['worse_cases_char_split_rate']:.1f}%)")
    print(f"  Better Cases: {char_stats['better_cases_with_char_splits']}/{undertrained_stats['better_cases_count']} ({char_stats['better_cases_char_split_rate']:.1f}%)")
    print(f"  Same Cases: {char_stats['same_cases_with_char_splits']}/{undertrained_stats['same_cases_count']} ({char_stats['same_cases_char_split_rate']:.1f}%)")
    
    print(f"\nSingle Character Tokens:")
    print(f"  Worse Cases - Avg: {single_char_stats['worse_avg']:.2f}")
    print(f"  Better Cases - Avg: {single_char_stats['better_avg']:.2f}")
    print(f"  Same Cases - Avg: {single_char_stats['same_avg']:.2f}")
    
    print(f"\nAverage Token Length:")
    print(f"  Worse Cases - Avg: {avg_len_stats['worse_avg']:.2f}")
    print(f"  Better Cases - Avg: {avg_len_stats['better_avg']:.2f}")
    print(f"  Same Cases - Avg: {avg_len_stats['same_avg']:.2f}")
    
    print(f"\nHigh Fragmentation Cases (>5 tokens):")
    print(f"  Worse Cases: {frag_stats['worse_cases_count']}")
    print(f"  Better Cases: {frag_stats['better_cases_count']}")
    print(f"  Same Cases: {frag_stats['same_cases_count']}")
    
    for category, examples in [('Worse', frag_stats['worse_examples']), 
                               ('Better', frag_stats['better_examples']), 
                               ('Same', frag_stats['same_examples'])]:
        if examples:
            print(f"\nExample High-Fragmentation {category} Cases:")
            for i, example in enumerate(examples, 1):
                print(f"  {i}. Subject: {example['subject']}, Question {example['dataset_index']}: {example['undertrained_token_count']} tokens, Score diff: {example['score_difference']:.4f}")
                print(f"     Tokens: {' '.join(example['undertrained_tokens'])}")
                print(f"     Answer: {example['alternative_predicted']} (Correct: {example['correct_answer']})")
    
    print(f"\n" + "-" * 50)
    print("CONCLUSIONS:")
    
    if abs(worse_vs_better_diff) > 0.5:
        direction = "MORE" if worse_vs_better_diff > 0 else "FEWER"
        print(f"✓ Worse cases tend to have {direction} undertrained tokens than Better cases ({worse_vs_better_diff:+.2f})")
    else:
        print(f"✗ No significant difference between Worse and Better cases ({worse_vs_better_diff:+.2f})")
    
    if abs(worse_vs_same_diff) > 0.5:
        direction = "MORE" if worse_vs_same_diff > 0 else "FEWER"
        print(f"✓ Worse cases tend to have {direction} undertrained tokens than Same cases ({worse_vs_same_diff:+.2f})")
    else:
        print(f"✗ No significant difference between Worse and Same cases ({worse_vs_same_diff:+.2f})")
    
    worse_better_char_diff = char_stats['worse_cases_char_split_rate'] - char_stats['better_cases_char_split_rate']
    worse_same_char_diff = char_stats['worse_cases_char_split_rate'] - char_stats['same_cases_char_split_rate']
    
    if worse_better_char_diff > 10:
        print(f"✓ Worse cases have higher character-level splitting than Better cases (+{worse_better_char_diff:.1f}%)")
    elif worse_same_char_diff > 10:
        print(f"✓ Worse cases have higher character-level splitting than Same cases (+{worse_same_char_diff:.1f}%)")
    else:
        print(f"✗ No significant difference in character-level splitting")
    
    worse_better_single_diff = single_char_stats['worse_avg'] - single_char_stats['better_avg']
    worse_same_single_diff = single_char_stats['worse_avg'] - single_char_stats['same_avg']
    
    if worse_better_single_diff > 0.5:
        print(f"✓ Worse cases have more single-character tokens than Better cases (+{worse_better_single_diff:.2f})")
    elif worse_same_single_diff > 0.5:
        print(f"✓ Worse cases have more single-character tokens than Same cases (+{worse_same_single_diff:.2f})")
    else:
        print(f"✗ No significant difference in single-character tokens")

def create_visualizations(comparison_data: Dict, save_plots: bool = False, output_dir: str = '.') -> None:
    detailed = comparison_data['detailed_results']
    summary = comparison_data['summary']
    df = pd.DataFrame(detailed)
    
    plt.style.use('default')
    
    # Figure 1: Distribution of Results (Pie Chart)
    plt.figure(figsize=(8, 6))
    labels = ['Better', 'Worse', 'Same']
    sizes = [summary['better_count'], summary['worse_count'], summary['same_count']]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Results', fontsize=14, fontweight='bold')
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'distribution_results.png'), dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to: {os.path.join(output_dir, 'distribution_results.png')}")
    plt.show()
    
    # Figure 2: Score Difference Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['score_difference'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
    plt.xlabel('Score Difference (Alternative - Baseline)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Score Differences', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'score_difference_distribution.png'), dpi=300, bbox_inches='tight')
        print(f"Score difference plot saved to: {os.path.join(output_dir, 'score_difference_distribution.png')}")
    plt.show()
    
    # Figure 3: Subject Performance Comparison
    subject_stats = analyze_subject_performance(comparison_data)
    subjects = list(subject_stats.keys())
    better_pcts = [subject_stats[s]['better_pct'] for s in subjects]
    worse_pcts = [subject_stats[s]['worse_pct'] for s in subjects]
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(subjects))
    width = 0.35
    
    plt.bar(x - width/2, better_pcts, width, label='Better %', color='green', alpha=0.7)
    plt.bar(x + width/2, worse_pcts, width, label='Worse %', color='red', alpha=0.7)
    
    plt.xlabel('Subjects')
    plt.ylabel('Percentage')
    plt.title('Performance by Subject: Better vs Worse Cases', fontsize=14, fontweight='bold')
    plt.xticks(x, subjects, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'subject_performance.png'), dpi=300, bbox_inches='tight')
        print(f"Subject performance plot saved to: {os.path.join(output_dir, 'subject_performance.png')}")
    plt.show()
    
    # Figure 4: Token Count Distribution - Only Better vs Worse Cases
    plt.figure(figsize=(12, 6))
    worse_tokens = df[df['improvement'] == 'Worse']['undertrained_token_count']
    better_tokens = df[df['improvement'] == 'Better']['undertrained_token_count']
    
    plt.hist([worse_tokens, better_tokens], bins=20, alpha=0.7, 
             label=['Worse Cases', 'Better Cases'], 
             color=['red', 'green'])
    plt.xlabel('Undertrained Token Count')
    plt.ylabel('Frequency')
    plt.title('Token Count Distribution: Better vs Worse Cases Only', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if len(worse_tokens) > 0 and len(better_tokens) > 0:
        worse_mean = np.mean(worse_tokens)
        better_mean = np.mean(better_tokens)
        plt.text(0.02, 0.98, f'Worse Cases - Mean: {worse_mean:.2f}\nBetter Cases - Mean: {better_mean:.2f}\nDifference: {worse_mean - better_mean:+.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'token_count_distribution.png'), dpi=300, bbox_inches='tight')
        print(f"Token count distribution saved to: {os.path.join(output_dir, 'token_count_distribution.png')}")
    plt.show()
    
    # Figure 5: Token Count vs Score Difference Scatter Plot
    plt.figure(figsize=(12, 8))
    colors_map = {'Better': 'green', 'Worse': 'red', 'Same': 'gray'}
    for improvement in ['Better', 'Worse', 'Same']:
        subset = df[df['improvement'] == improvement]
        plt.scatter(subset['undertrained_token_count'], subset['score_difference'], 
                   alpha=0.6, label=improvement, color=colors_map[improvement], s=50)
    
    plt.xlabel('Undertrained Token Count')
    plt.ylabel('Score Difference')
    plt.title('Token Count vs Score Difference', fontsize=14, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    correlation = np.corrcoef(df['undertrained_token_count'], df['score_difference'])[0, 1]
    plt.text(0.05, 0.95, f'Overall Correlation: {correlation:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=12)
    
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'token_count_vs_score_difference.png'), dpi=300, bbox_inches='tight')
        print(f"Token count scatter plot saved to: {os.path.join(output_dir, 'token_count_vs_score_difference.png')}")
    plt.show()
    
    # Figure 6: Token Count Difference vs Score Difference Scatter Plot
    plt.figure(figsize=(12, 8))
    colors_map = {'Better': 'green', 'Worse': 'red', 'Same': 'gray'}
    for improvement in ['Better', 'Worse', 'Same']:
        subset = df[df['improvement'] == improvement]
        plt.scatter(subset['token_count_difference'], subset['score_difference'], 
                   alpha=0.6, label=improvement, color=colors_map[improvement], s=50)
    
    plt.xlabel('Token Count Difference (Alternative - Baseline)')
    plt.ylabel('Score Difference (Alternative - Baseline)')
    plt.title('Token Count Difference vs Score Difference', fontsize=14, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    correlation_token_diff = np.corrcoef(df['token_count_difference'], df['score_difference'])[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation_token_diff:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=12)
    
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'token_count_difference_vs_score_difference.png'), dpi=300, bbox_inches='tight')
        print(f"Token count difference scatter plot saved to: {os.path.join(output_dir, 'token_count_difference_vs_score_difference.png')}")
    plt.show()

def save_detailed_results_to_csv(comparison_data: Dict, output_file: str) -> None:
    df = pd.DataFrame(comparison_data['detailed_results'])
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Compare baseline and alternative tokenization results for MMLU',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tokenization_mc.py -b baseline.json -a alternative.json
  python tokenization_mc.py -b baseline.json -a alternative.json --no-plots
  python tokenization_mc.py -b baseline.json -a alternative.json -o results --save-plots
  python tokenization_mc.py -b baseline.json -a alternative.json --csv-only
        """
    )
    
    parser.add_argument('-b', '--baseline', type=str, required=True,
                       help='Path to baseline tokenization results JSON file')
    parser.add_argument('-a', '--alternative', type=str, required=True,
                       help='Path to alternative tokenization results JSON file')
    parser.add_argument('-o', '--output', type=str, default='mmlu_tokenization_comparison',
                       help='Output filename prefix (default: mmlu_tokenization_comparison)')
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
            print("Loading and comparing MMLU tokenization results...")
        
        comparison_data = compare_tokenization_results(args.baseline, args.alternative)
        
        if args.csv_only:
            csv_path = os.path.join(args.output_dir, f"{args.output}.csv")
            save_detailed_results_to_csv(comparison_data, csv_path)
        else:
            if not args.quiet:
                generate_analysis_report(comparison_data)
                
                undertrained_analysis = analyze_undertrained_impact(comparison_data)
                print_undertrained_analysis(undertrained_analysis)
                
                same_score_analysis = analyze_same_score_tokenization(comparison_data)
                print_same_score_analysis(same_score_analysis)
            
            if not args.no_csv:
                csv_path = os.path.join(args.output_dir, f"{args.output}.csv")
                save_detailed_results_to_csv(comparison_data, csv_path)
            
            if not args.no_plots:
                create_visualizations(comparison_data, args.save_plots, args.output_dir)
        
        if not args.quiet:
            print(f"\nMMLU analysis completed successfully!")
            
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