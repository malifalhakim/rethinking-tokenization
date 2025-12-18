import os
import sys
import argparse
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import torch
import gc

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sacrebleu.metrics import BLEU

from quantifier.trainness.magikarp import TokenNorm
from tokenizer.bpe_norm_tokenizer import BPENormTokenizer
from utils.helper import (
    prepare_model, 
    process_prompt, 
    generate_response,
    find_optimal_batch_size,
    initiate_seed
)

TRANSLATION_PROMPT_TEMPLATE = """Translate the following text to {target_language}:

{source_text}

Translation:"""


def load_csv_dataset(csv_path: str, limit: int = None) -> pd.DataFrame:
    """Load dataset from CSV file."""
    df = pd.read_csv(csv_path)
    
    required_cols = ["original_text", "translated_text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: {col}")
    
    if limit is not None:
        df = df.head(limit)
    
    return df


def apply_translation_prompt(source_text: str, target_language: str) -> str:
    """Format source text into translation prompt."""
    return TRANSLATION_PROMPT_TEMPLATE.format(
        target_language=target_language,
        source_text=source_text
    )


def get_tokenized_strings(tokenizer, tokenized_prompts, use_vllm: bool) -> list[list[str]]:
    """Convert tokenized prompts to list of token strings."""
    if use_vllm:
        tokenized_strings = []
        for token_ids in tokenized_prompts:
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            tokenized_strings.append(tokens)
        return tokenized_strings
    else:
        tokenized_strings = []
        for i in range(tokenized_prompts["input_ids"].shape[0]):
            tokens = tokenizer.convert_ids_to_tokens(
                tokenized_prompts["input_ids"][i].tolist()
            )
            tokenized_strings.append(tokens)
        return tokenized_strings


def evaluate_batch(
    model,
    tokenizer,
    batch_data: list[dict],
    target_language: str,
    use_vllm: bool,
    max_new_tokens: int,
    seed: int,
    collect_tokenized: bool = True
) -> list[dict]:
    """Evaluate a batch of translation samples."""
    
    prompts = [
        apply_translation_prompt(item["original_text"], target_language)
        for item in batch_data
    ]
    
    tokenized_prompts = process_prompt(tokenizer, prompts, use_vllm)
    if args.use_vllm:
        tokenized_prompts = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    
    tokenized_strings = None
    if collect_tokenized:
        tokenized_strings = get_tokenized_strings(tokenizer, tokenized_prompts, use_vllm)
    
    initiate_seed(seed)
    responses = generate_response(
        model, tokenized_prompts, tokenizer, use_vllm, 
        max_new_tokens=max_new_tokens, seed=seed
    )
    
    results = []
    for idx, (item, response) in enumerate(zip(batch_data, responses)):
        result = {
            "original_text": item["original_text"],
            "reference": item["translated_text"],
            "prediction": response.strip(),
        }
        
        if collect_tokenized and tokenized_strings:
            result["tokenized_input"] = tokenized_strings[idx]
        
        results.append(result)
    
    return results


def calculate_bleu_scores(results: list[dict]) -> dict:
    """Calculate SacreBLEU scores for the results."""
    bleu = BLEU()
    
    predictions = [r["prediction"] for r in results]
    references = [[r["reference"]] for r in results]
    
    corpus_score = bleu.corpus_score(predictions, references)
    
    sentence_scores = []
    for pred, ref in zip(predictions, references):
        score = bleu.sentence_score(pred, ref)
        sentence_scores.append(score.score)
    
    return {
        "corpus_bleu": corpus_score.score,
        "corpus_bleu_details": {
            "score": corpus_score.score,
            "counts": corpus_score.counts,
            "totals": corpus_score.totals,
            "precisions": corpus_score.precisions,
            "bp": corpus_score.bp,
            "sys_len": corpus_score.sys_len,
            "ref_len": corpus_score.ref_len,
        },
        "sentence_scores": sentence_scores,
        "avg_sentence_bleu": sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0.0
    }


def save_results(logs: list, stats: dict, output_path: str):
    """Save evaluation results to a JSON file."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "statistics": stats,
        "detailed_logs": logs,
    }
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")


def main(args):
    # -- Load dataset --
    print(f"Loading dataset from: {args.csv_path}")
    df = load_csv_dataset(args.csv_path, limit=args.limit)
    print(f"Loaded {len(df)} samples")
    
    # -- Prepare model and tokenizer --
    print(f"Loading model: {args.model_name}")
    
    vllm_kwargs = {}
    hf_kwargs = {}
    
    if args.use_vllm:
        if args.gpu_memory_utilization:
            vllm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
        if args.tensor_parallel_size:
            vllm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
        model, tokenizer = prepare_model(args.model_name, use_vllm=True, **vllm_kwargs)
    else:
        model, tokenizer = prepare_model(args.model_name, use_vllm=False, **hf_kwargs)
        model = model.to(args.device)
    
    # -- Apply custom tokenizer if specified --
    if args.tokenizer_type == "norm":
        if not args.magikarp_path:
            raise ValueError("--magikarp_path is required when using --tokenizer_type norm")
        print(f"Loading TokenNorm from: {args.magikarp_path}")
        token_norm = TokenNorm(args.magikarp_path, tokenizer)
        tokenizer = BPENormTokenizer(tokenizer, token_norm)
    
    # -- Find optimal batch size --
    sample_prompts = [
        apply_translation_prompt(text, args.target_language)
        for text in df["original_text"].head(min(10, len(df))).tolist()
    ]
    
    optimal_batch_size = find_optimal_batch_size(
        model, tokenizer, sample_prompts,
        use_vllm=args.use_vllm,
        max_new_tokens=args.max_new_tokens,
        start_batch_size=args.batch_size
    )
    optimal_batch_size = df.shape[0] if args.use_vllm else optimal_batch_size
    print(f"Using batch size: {optimal_batch_size}")
    
    # -- Prepare data as list of dicts --
    data_list = df.to_dict("records")
    
    # -- Evaluate in batches --
    all_results = []
    
    for i in tqdm(range(0, len(data_list), optimal_batch_size), desc="Evaluating"):
        batch_data = data_list[i:i + optimal_batch_size]
        
        batch_results = evaluate_batch(
            model=model,
            tokenizer=tokenizer,
            batch_data=batch_data,
            target_language=args.target_language,
            use_vllm=args.use_vllm,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            collect_tokenized=args.save_tokenized
        )
        
        all_results.extend(batch_results)
        
        if not args.use_vllm:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # -- Calculate BLEU scores --
    print("\nCalculating BLEU scores...")
    bleu_scores = calculate_bleu_scores(all_results)
    
    # -- Add sentence-level BLEU to each result --
    for idx, result in enumerate(all_results):
        result["sentence_bleu"] = bleu_scores["sentence_scores"][idx]
    
    # -- Prepare statistics --
    stats = {
        "model_name": args.model_name,
        "tokenizer_type": args.tokenizer_type or "standard",
        "target_language": args.target_language,
        "total_samples": len(all_results),
        "corpus_bleu": bleu_scores["corpus_bleu"],
        "avg_sentence_bleu": bleu_scores["avg_sentence_bleu"],
        "bleu_details": bleu_scores["corpus_bleu_details"],
    }
    
    print(f"\n{'='*50}")
    print(f"Corpus BLEU Score: {bleu_scores['corpus_bleu']:.4f}")
    print(f"Average Sentence BLEU: {bleu_scores['avg_sentence_bleu']:.4f}")
    print(f"Total Samples: {len(all_results)}")
    print(f"{'='*50}")
    
    # -- Save results --
    if args.output_path:
        save_results(all_results, stats, args.output_path)
    
    return stats, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on machine translation dataset")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV file with 'original_text' and 'translated_text' columns"
    )
    parser.add_argument(
        "--target_language",
        type=str,
        default="English",
        help="Target language for translation (default: English)"
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["norm", "standard"],
        default=None,
        help="Type of tokenizer to use"
    )
    parser.add_argument(
        "--magikarp_path",
        type=str,
        default=None,
        help="Path to Magikarp JSONL file for token normalization"
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for faster inference"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save evaluation results (JSON)"
    )
    parser.add_argument(
        "--save_tokenized",
        action="store_true",
        help="Save tokenized inputs in the logs"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="GPU memory utilization for vLLM (0.0-1.0)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Tensor parallel size for vLLM"
    )
    
    args = parser.parse_args()
    main(args)