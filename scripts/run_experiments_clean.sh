#!/bin/bash
# Run evaluation experiments independently

# ==============================
# Customizable variables
# ==============================
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

# Quantifier files
NORM_QUANTIFIER="results/undertrained/l2-norm/Qwen_Qwen2_5_7B_Instruct.jsonl"
ENTROPY_QUANTIFIER="results/undertrained/entropy/Qwen2.5-7B-Instruct_token_entropy.json"

# ==============================
# Experiments
# ==============================

python evaluation/mmlu.py \
  --model_name "$MODEL_NAME" \
  --use_alternative_tokenizer \
  --type norm \
  --device cuda:1 \
  --num_tokenizations_samples 1 \
  --quantifier_file "$NORM_QUANTIFIER"

python evaluation/mmlu.py \
  --model_name "$MODEL_NAME" \
  --use_alternative_tokenizer \
  --type entropy \
  --device cuda:1 \
  --num_tokenizations_samples 1 \
  --quantifier_file "$ENTROPY_QUANTIFIER"

python evaluation/wmt.py \
  --model_name "$MODEL_NAME" \
  --use_alternative_tokenizer \
  --type norm \
  --num_tokenizations_samples 1 \
  --quantifier_file "$NORM_QUANTIFIER" \
  --device cuda:1

python evaluation/wmt.py \
  --model_name "$MODEL_NAME" \
  --device cuda:1 \
  --use_alternative_tokenizer \
  --type entropy \
  --num_tokenizations_samples 1 \
  --quantifier_file "$ENTROPY_QUANTIFIER"
