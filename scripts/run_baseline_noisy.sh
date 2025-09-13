#!/bin/bash
# Run evaluation experiments independently

# ==============================
# Customizable variables
# ==============================
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME="Amadeus99/amazon-reviews-qwen_qwen2_5_7b_instruct"
DATASET_PATH="dataset/MTNT/test.en-fr.csv"

# ==============================
# Experiments
# ==============================

python evaluation/amazon_reviews.py \
  --model_name "$MODEL_NAME" \
  --dataset_name "$DATASET_NAME"

python evaluation/mtnt.py \
  --model_name "$MODEL_NAME" \
  --dataset_path "$DATASET_PATH"