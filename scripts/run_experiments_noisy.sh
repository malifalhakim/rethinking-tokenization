#!/bin/bash
# Run evaluation experiments independently

# ==============================
# Customizable variables
# ==============================
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME="Amadeus99/amazon-reviews-qwen_qwen2_5_7b_instruct"
DATASET_PATH="dataset/MTNT/test.en-fr.csv"

# Quantifier files
NORM_QUANTIFIER="results/undertrained/l2-norm/Qwen_Qwen2_5_7B_Instruct.jsonl"
ENTROPY_QUANTIFIER="results/undertrained/entropy/Qwen2.5-7B-Instruct_token_entropy.json"
ENTROPY_UNDERTRAINED_FILE="results/undertrained/entropy/Qwen2.5-7B-Instruct_glitch_tokens.pkl"

# ==============================
# Experiments
# ==============================

python evaluation/amazon_reviews.py \
  --model_name "$MODEL_NAME" \
  --device cuda:1 \
  --dataset_name "$DATASET_NAME" \
  --use_alternative_tokenizer \
  --type norm \
  --quantifier_file "$NORM_QUANTIFIER"

python evaluation/amazon_reviews.py \
  --model_name "$MODEL_NAME" \
  --device cuda:1 \
  --dataset_name "$DATASET_NAME" \
  --use_alternative_tokenizer \
  --type entropy \
  --quantifier_file "$ENTROPY_QUANTIFIER" \
  --undertrained_entropy_file "$ENTROPY_UNDERTRAINED_FILE"

python evaluation/mtnt.py \
  --model_name "$MODEL_NAME" \
  --device cuda:1 \
  --dataset_path "$DATASET_PATH" \
  --use_alternative_tokenizer \
  --type norm \
  --quantifier_file "$NORM_QUANTIFIER"

python evaluation/mtnt.py \
  --model_name "$MODEL_NAME" \
  --device cuda:1 \
  --dataset_path "$DATASET_PATH" \
  --use_alternative_tokenizer \
  --type entropy \
  --quantifier_file "$ENTROPY_QUANTIFIER" \
  --undertrained_entropy_file "$ENTROPY_UNDERTRAINED_FILE"
