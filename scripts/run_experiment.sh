#!/bin/bash

# =============================================================================
# MAIN CONFIGURATION (CHANGE THIS SECTION)
# =============================================================================
IS_TESTING=${IS_TESTING:-false}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B-Instruct"}
MAGIKARP_PATH=${MAGIKARP_PATH:-"results/undertrained/l2-norm/Qwen_Qwen2_5_7B_Instruct.jsonl"}
TOKENIZER_TYPE=${TOKENIZER_TYPE:-"norm"}

SANITIZED_MODEL_NAME="${MODEL_NAME//\//_}"
SANITIZED_MODEL_NAME="${SANITIZED_MODEL_NAME//./_}"
SANITIZED_MODEL_NAME="${SANITIZED_MODEL_NAME//-/_}"

# Experiments to run (set to true/false)
RUN_PASSKEY=${RUN_PASSKEY:-true}
RUN_GSM8K=${RUN_GSM8K:-true}
RUN_MMLU=${RUN_MMLU:-true}
RUN_MTNT=${RUN_MTNT:-true}

# =============================================================================
# HELPER FUNCTIONS (DO NOT MODIFY)
# =============================================================================
send_notification() {
    local message=$1
    curl -d "$message" ntfy.sh/rethinking-tokenization
}

get_limit() {
    local default_limit=$1
    if [ "$IS_TESTING" = true ]; then
        echo 5
    else
        echo "$default_limit"
    fi
}

# =============================================================================
# EXPERIMENT: PASSKEY RETRIEVAL
# =============================================================================
run_passkey_retrieval() {
    local use_vllm=true
    local number_of_data=$(get_limit 500)
    local output_dir="results/experiments/passkey_retrieval/$SANITIZED_MODEL_NAME"
    local output_path="$output_dir/$TOKENIZER_TYPE.jsonl"
    local stats_path="$output_dir/stats_$TOKENIZER_TYPE.json"
    local log_path="$output_dir/log_$TOKENIZER_TYPE.txt"

    mkdir -p "$output_dir"
    echo "Logging to $log_path"

    local vllm_flag=""
    [ "$use_vllm" = true ] && vllm_flag="--use_vllm"

    local limit_flag=""
    [ -n "$number_of_data" ] && limit_flag="--number_of_data $number_of_data"

    python -u evaluation/passkey_retrieval.py \
        --model_name "$MODEL_NAME" \
        --magikarp_path "$MAGIKARP_PATH" \
        --tokenizer_type "$TOKENIZER_TYPE" \
        $limit_flag \
        --output_path "$output_path" \
        --stats_path "$stats_path" \
        $vllm_flag \
        2>&1 | tee "$log_path"

    send_notification "Passkey Retrieval Experiment for $MODEL_NAME with $TOKENIZER_TYPE tokenizer completed."
}

# =============================================================================
# EXPERIMENT: GSM8K EVALUATION
# =============================================================================
run_gsm8k() {
    local use_vllm=true
    local dataset_path="dataset/GSM8K/gsm8k_modified_dataset.jsonl"
    local limit=$(get_limit "")
    local output_dir="results/experiments/gsm8k/$SANITIZED_MODEL_NAME"
    local output_path="$output_dir/$TOKENIZER_TYPE.jsonl"
    local stats_path="$output_dir/stats_$TOKENIZER_TYPE.json"
    local log_path="$output_dir/log_$TOKENIZER_TYPE.txt"

    mkdir -p "$output_dir"
    echo "Logging to $log_path"

    local vllm_flag=""
    [ "$use_vllm" = true ] && vllm_flag="--use_vllm"

    local limit_flag=""
    [ -n "$limit" ] && limit_flag="--limit $limit"

    python -u evaluation/gsm8k.py \
        --model_name "$MODEL_NAME" \
        --magikarp_path "$MAGIKARP_PATH" \
        --tokenizer_type "$TOKENIZER_TYPE" \
        --dataset_path "$dataset_path" \
        $limit_flag \
        --detailed_output_path "$output_path" \
        --stats_output_path "$stats_path" \
        $vllm_flag \
        2>&1 | tee "$log_path"

    send_notification "GSM8K Evaluation Experiment for $MODEL_NAME with $TOKENIZER_TYPE tokenizer completed."
}

# =============================================================================
# EXPERIMENT: MMLU EVALUATION
# =============================================================================
run_mmlu() {
    local starting_batch_size=32
    local limit=$(get_limit "")
    local output_dir="results/experiments/mmlu/$SANITIZED_MODEL_NAME"
    local output_path="$output_dir/$TOKENIZER_TYPE.json"
    local log_path="$output_dir/log_$TOKENIZER_TYPE.txt"

    mkdir -p "$output_dir"
    echo "Logging to $log_path"

    local limit_flag=""
    [ -n "$limit" ] && limit_flag="--limit $limit"

    python -u evaluation/mmlu.py \
        --model_name "$MODEL_NAME" \
        --magikarp_path "$MAGIKARP_PATH" \
        --tokenizer_type "$TOKENIZER_TYPE" \
        --output_path "$output_path" \
        --batch_size "$starting_batch_size" \
        $limit_flag \
        --save_tokenized \
        2>&1 | tee "$log_path"

    send_notification "MMLU Evaluation Experiment for $MODEL_NAME with $TOKENIZER_TYPE tokenizer completed."
}

# =============================================================================
# EXPERIMENT: MTNT EVALUATION
# =============================================================================
run_mtnt() {
    local use_vllm=true
    local csv_path="dataset/MTNT/test.fr-en.csv"
    local target_language="English"
    local starting_batch_size=128
    local seed=42
    local limit=$(get_limit "")
    local output_dir="results/experiments/mtnt/$SANITIZED_MODEL_NAME"
    local output_path="$output_dir/$TOKENIZER_TYPE.json"
    local log_path="$output_dir/log_$TOKENIZER_TYPE.txt"

    mkdir -p "$output_dir"
    echo "Logging to $log_path"

    local vllm_flag=""
    [ "$use_vllm" = true ] && vllm_flag="--use_vllm"

    local limit_flag=""
    [ -n "$limit" ] && limit_flag="--limit $limit"

    python -u evaluation/mtnt.py \
        --model_name "$MODEL_NAME" \
        --magikarp_path "$MAGIKARP_PATH" \
        --tokenizer_type "$TOKENIZER_TYPE" \
        --csv_path "$csv_path" \
        --target_language "$target_language" \
        --output_path "$output_path" \
        --batch_size "$starting_batch_size" \
        --seed "$seed" \
        $limit_flag \
        $vllm_flag \
        --save_tokenized \
        2>&1 | tee "$log_path"

    send_notification "MTNT Evaluation Experiment for $MODEL_NAME with $TOKENIZER_TYPE tokenizer completed."
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================
[ "$RUN_PASSKEY" = true ] && run_passkey_retrieval
[ "$RUN_GSM8K" = true ] && run_gsm8k
[ "$RUN_MMLU" = true ] && run_mmlu
[ "$RUN_MTNT" = true ] && run_mtnt

send_notification "All experiments for $MODEL_NAME with $TOKENIZER_TYPE tokenizer completed."