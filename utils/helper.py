import torch
import random
import numpy as np
import gc

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from vllm import LLM, SamplingParams

def initiate_seed(seed: int):
    """
    Initiates the random seed for reproducibility.
    """
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_dataset(dataset_name: str, dataset_subset: str, dataset_split: str, prompt_col: str, limit: int = None):
    """
    Prepares the dataset for evaluation.
    """
    dataset = load_dataset(dataset_name, dataset_subset, split=dataset_split) if dataset_subset else load_dataset(dataset_name, split=dataset_split)
    prompts = [data[prompt_col] for data in dataset]
    if limit is not None:
        prompts = prompts[:limit]
    return prompts

def prepare_model(model_name: str, use_vllm: bool, **kwargs):
    """
    Prepares the model for evaluation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_vllm:
        model = LLM(model=model_name, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **kwargs)
        model.eval()

    return model, tokenizer

def process_prompt(tokenizer, prompts: list[str], use_vllm: bool):
    """
    Tokenizes the prompt.
    """
    if use_vllm:
        processed_prompts = []
        for prompt in prompts:
            chat = [{"role": "user", "content": prompt}]
            processed_prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
        return processed_prompts
    
    chats = [[{"role": "user", "content": p}] for p in prompts]
    templated_prompts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chats]

    tokenizer.padding_side = "left"
    tokenized_inputs = tokenizer(templated_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    return tokenized_inputs

def generate_response_with_params(model, tokenized_prompts, tokenizer, use_vllm: bool, max_new_tokens: int = 512, seed: int = 42, **generation_params):
    """
    Generates a response from the model with specific generation parameters.
    Only overrides the parameters explicitly provided.
    """
    if use_vllm:
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens, 
            seed=seed,
            **generation_params
        )
        if isinstance(tokenized_prompts[0], str):
            outputs = model.generate(tokenized_prompts, sampling_params)
        else:
            prompts_batch = [
                {"prompt_token_ids": ids} for ids in tokenized_prompts
            ]
            outputs = model.generate(prompts=prompts_batch, sampling_params=sampling_params)
        
        responses = [output.outputs[0].text for output in outputs]
    else:
        input_ids = tokenized_prompts["input_ids"].to(model.device)
        attention_mask = tokenized_prompts["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=tokenizer.pad_token_id,
                **generation_params
            )
      
        responses = []
        for i in range(len(outputs)):
            input_length = len(input_ids[i])
            generated_tokens = outputs[i][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)

    return responses

def generate_response(model, tokenized_prompts, tokenizer, use_vllm: bool, max_new_tokens: int = 512, seed: int = 42):
    """
    Generates a response from the model with default parameters.
    """
    return generate_response_with_params(model, tokenized_prompts, tokenizer, use_vllm, max_new_tokens, seed)

def generate_single_seed_responses(model, tokenized_prompts, tokenizer, use_vllm: bool, max_new_tokens: int = 512, seed: int = 42):
    """
    Generates responses with a single seed for each prompt.
    """
    initiate_seed(seed)
    responses = generate_response(model, tokenized_prompts, tokenizer, use_vllm, max_new_tokens, seed)
    
    return [{f"seed_{seed}": response} for response in responses]

def generate_multiple_seed_responses(model, tokenized_prompts, tokenizer, use_vllm: bool, max_new_tokens: int = 512, seeds: list = [42, 123, 456]):
    """
    Generates responses with multiple seeds for each prompt.
    Returns a list of dicts, where each dict contains responses for different seeds.
    """
    all_responses = []
    
    for seed in seeds:
        initiate_seed(seed)
        responses = generate_response(model, tokenized_prompts, tokenizer, use_vllm, max_new_tokens, seed)
        all_responses.append(responses)
    
    grouped_responses = []
    for prompt_responses in zip(*all_responses):
        response_dict = {f"seed_{seed}": response for seed, response in zip(seeds, prompt_responses)}
        grouped_responses.append(response_dict)
    
    return grouped_responses


def process_batch(model, tokenizer, batch_prompts, use_vllm, max_new_tokens, generation_config):
    """
    Processes a single batch of prompts and returns log entries.
    """
    tokenized_batch = process_prompt(tokenizer, batch_prompts, use_vllm)
    
    if generation_config["type"] == "single_seed":
        batch_responses = generate_single_seed_responses(
            model, tokenized_batch, tokenizer, use_vllm,
            max_new_tokens=max_new_tokens,
            seed=generation_config["seed"]
        )
    elif generation_config["type"] == "multi_seed":
        batch_responses = generate_multiple_seed_responses(
            model, tokenized_batch, tokenizer, use_vllm,
            max_new_tokens=max_new_tokens,
            seeds=generation_config["seeds"]
        )
    else:
        raise ValueError(f"Unknown generation type: {generation_config['type']}")
    
    log_entries = []
    for prompt, responses in zip(batch_prompts, batch_responses):
        log_entry = {
            "prompt": prompt,
            "responses": responses,
            "generation_config": generation_config
        }
        log_entries.append(log_entry)
    
    return log_entries

def get_generation_config(is_multi_seed: bool):
    """
    Determines the generation configuration based on arguments.
    """
    if is_multi_seed:
        return {
            "type": "multi_seed",
            "seeds": [42, 123, 456]
        }
    else:
        return {
            "type": "single_seed",
            "seed": 42
        }

def find_optimal_batch_size(model, tokenizer, sample_prompts: list[str], use_vllm: bool, max_new_tokens: int = 512, start_batch_size: int = 64):
    """
    Attempts to find the largest batch size that fits in memory by halving the size upon OOM errors.
    For vLLM, it simply returns the start_batch_size as vLLM manages memory dynamically.
    """
    if use_vllm:
        print(f"Using vLLM: Returning default batch size {start_batch_size} (vLLM manages memory dynamically).")
        return start_batch_size

    print(f"Finding optimal batch size starting from {start_batch_size}...")
    batch_size = start_batch_size
    test_prompts = sample_prompts * (start_batch_size // len(sample_prompts) + 1)

    while batch_size > 0:
        try:
            current_prompts = test_prompts[:batch_size]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            print(f"Testing batch size: {batch_size}")
            
            config = get_generation_config(is_multi_seed=False)

            process_batch(
                model, tokenizer, current_prompts, False, max_new_tokens=max_new_tokens, generation_config=config
            )
            
            print(f"Batch size {batch_size} successful.")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM detected with batch size {batch_size}. Halving...")
                batch_size //= 2
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            else:
                raise

def find_sequence_index(full_list: list, sub_list: list) -> int:
    """
    Finds the starting index of sub_list in full_list.
    Returns -1 if sub_list is not found.
    """
    sub_len = len(sub_list)
    for i in range(len(full_list) - sub_len + 1):
        if full_list[i:i + sub_len] == sub_list:
            return i
    return -1

def inject_token_at_placeholder(input_ids: list, placeholder_ids: list, token_ids: list) -> list:
    """
    Replaces the placeholder_ids in input_ids with token_ids.
    """
    index = find_sequence_index(input_ids, placeholder_ids)
    if index == -1:
        raise ValueError("Placeholder IDs not found in input IDs.")
    
    new_input_ids = (
        input_ids[:index] + 
        token_ids + 
        input_ids[index + len(placeholder_ids):]
    )
    return new_input_ids