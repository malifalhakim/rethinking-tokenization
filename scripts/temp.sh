#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
MAGIKARP_PATH="results/undertrained/l2-norm/Qwen_Qwen2_5_7B_Instruct.jsonl" \
ENTROPY_PATH="results/undertrained/entropy/Qwen2.5-7B-Instruct_token_entropy.json" \
ENTROPY_PKL="results/undertrained/entropy/Qwen2.5-7B-Instruct_glitch_tokens.pkl" \
TOKENIZER_TYPE="standard" \
QUANTIFIER_TYPE="entropy" \
RUN_MMLU=false \
./scripts/run_experiment.sh

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
MAGIKARP_PATH="results/undertrained/l2-norm/Qwen_Qwen2_5_7B_Instruct.jsonl" \
ENTROPY_PATH="results/undertrained/entropy/Qwen2.5-7B-Instruct_token_entropy.json" \
ENTROPY_PKL="results/undertrained/entropy/Qwen2.5-7B-Instruct_glitch_tokens.pkl" \
TOKENIZER_TYPE="entropy" \
QUANTIFIER_TYPE="entropy" \
RUN_MMLU=false \
./scripts/run_experiment.sh

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
MAGIKARP_PATH="results/undertrained/l2-norm/meta_llama_Llama_3_1_8B_Instruct.jsonl" \
ENTROPY_PATH="results/undertrained/entropy/Llama_3_1_8B_Instruct_token_entropy.json" \
ENTROPY_PKL="results/undertrained/entropy/Llama_3_1_8B_Instruct_glitch_tokens.pkl" \
TOKENIZER_TYPE="standard" \
QUANTIFIER_TYPE="entropy" \
RUN_MMLU=false \
./scripts/run_experiment.sh

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
MAGIKARP_PATH="results/undertrained/l2-norm/meta_llama_Llama_3_1_8B_Instruct.jsonl" \
ENTROPY_PATH="results/undertrained/entropy/Llama_3_1_8B_Instruct_token_entropy.json" \
ENTROPY_PKL="results/undertrained/entropy/Llama_3_1_8B_Instruct_glitch_tokens.pkl" \
TOKENIZER_TYPE="entropy" \
QUANTIFIER_TYPE="entropy" \
RUN_MMLU=false \
./scripts/run_experiment.sh

# ----------------------------------#
# --- RUN MMLU EXPERIMENTS ONLY --- #
# ----------------------------------#

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
MAGIKARP_PATH="results/undertrained/l2-norm/Qwen_Qwen2_5_7B_Instruct.jsonl" \
ENTROPY_PATH="results/undertrained/entropy/Qwen2.5-7B-Instruct_token_entropy.json" \
ENTROPY_PKL="results/undertrained/entropy/Qwen2.5-7B-Instruct_glitch_tokens.pkl" \
TOKENIZER_TYPE="standard" \
QUANTIFIER_TYPE="entropy" \
RUN_PASSKEY=false \
RUN_GSM8K=false \
RUN_MTNT=false \
./scripts/run_experiment.sh

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
MAGIKARP_PATH="results/undertrained/l2-norm/Qwen_Qwen2_5_7B_Instruct.jsonl" \
ENTROPY_PATH="results/undertrained/entropy/Qwen2.5-7B-Instruct_token_entropy.json" \
ENTROPY_PKL="results/undertrained/entropy/Qwen2.5-7B-Instruct_glitch_tokens.pkl" \
TOKENIZER_TYPE="entropy" \
QUANTIFIER_TYPE="entropy" \
RUN_PASSKEY=false \
RUN_GSM8K=false \
RUN_MTNT=false \
./scripts/run_experiment.sh

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
MAGIKARP_PATH="results/undertrained/l2-norm/meta_llama_Llama_3_1_8B_Instruct.jsonl" \
ENTROPY_PATH="results/undertrained/entropy/Llama_3_1_8B_Instruct_token_entropy.json" \
ENTROPY_PKL="results/undertrained/entropy/Llama_3_1_8B_Instruct_glitch_tokens.pkl" \
TOKENIZER_TYPE="standard" \
QUANTIFIER_TYPE="entropy" \
RUN_PASSKEY=false \
RUN_GSM8K=false \
RUN_MTNT=false \
./scripts/run_experiment.sh

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
MAGIKARP_PATH="results/undertrained/l2-norm/meta_llama_Llama_3_1_8B_Instruct.jsonl" \
ENTROPY_PATH="results/undertrained/entropy/Llama_3_1_8B_Instruct_token_entropy.json" \
ENTROPY_PKL="results/undertrained/entropy/Llama_3_1_8B_Instruct_glitch_tokens.pkl" \
TOKENIZER_TYPE="entropy" \
QUANTIFIER_TYPE="entropy" \
RUN_PASSKEY=false \
RUN_GSM8K=false \
RUN_MTNT=false \
./scripts/run_experiment.sh