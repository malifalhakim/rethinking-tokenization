import argparse
import json
import tqdm
import pickle

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from glitchminer import GlitchMiner
from glitchminer.llm_template import get_template_for_model
from glitchminer.tokenfilter import TokenFilter

def entropy(probs):
    """Calculates the entropy of a probability distribution."""
    return -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)

def setup_model_and_tokenizer(model_name, device_arg=None):
    """Loads the model and tokenizer and sets the device."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if device_arg:
        if device_arg.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device_arg)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer


def calculate_all_token_entropy(model, tokenizer, no_need_tokens, chat_template=None):
    """
    Calculates the entropy for all tokens in the vocabulary.
    """
    device = model.device
    vocab_size = model.get_input_embeddings().weight.shape[0]

    # --- Create a mask to exclude unwanted tokens ---
    mask = torch.ones(vocab_size, dtype=torch.bool, device="cpu")
    mask[no_need_tokens] = False
    valid_token_ids = torch.where(mask)[0]

    # --- Get the appropriate chat template for the model ---
    if chat_template is None:
        chat_template = get_template_for_model(model.config._name_or_path)

    system_format = chat_template.system_format
    user_format = chat_template.user_format
    assistant_prefill = ' Sure, the string is: "«'
    system_message = ''

    all_entropy_data = []

    model.eval()
    with torch.no_grad():
        for token_id in tqdm.tqdm(valid_token_ids, desc="Calculating Token Entropy"):
            token_id_item = token_id.item()
            try:
                token = tokenizer.decode([token_id_item])
            except Exception as e:
                print(f"Could not decode token ID {token_id_item}: {e}")
                token = "[DECODING_ERROR]"

            # --- Construct the input prompt ---
            user_prompt = f'Please repeat the string: "«{token}»"'
            formatted_input = (system_format.format(content=system_message) if system_format else "") + user_format.format(content=user_prompt) + assistant_prefill
            input_ids = tokenizer.encode(formatted_input, return_tensors="pt").to(device)

            # --- Get model output and calculate entropy ---
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            entropy_value = entropy(next_token_probs).item()

            all_entropy_data.append({"token_id": token_id_item, "token": token, "entropy": entropy_value})
   
    return all_entropy_data

def get_glitch_tokens(model, tokenizer, num_iterations:int=125, batch_size:int=8, k:int=32, if_print:bool=True):
    glitch_tokens, glitch_token_ids = GlitchMiner(
        model,
        tokenizer,
        num_iterations=num_iterations,
        batch_size=batch_size,
        k=k,
        if_print=if_print,
        print_language="ENG"
    )

    return glitch_tokens, glitch_token_ids

def get_all_entropy_data(model, tokenizer, chat_template=None):
    print("Filtering out special tokens...")
    token_filter = TokenFilter(model, tokenizer)
    skip_tokens = token_filter.filter_token()
    print(f"Found {len(skip_tokens)} tokens to skip.")

    all_entropy_data = calculate_all_token_entropy(
        model,
        tokenizer,
        no_need_tokens=skip_tokens,
        chat_template=chat_template
    )
    return all_entropy_data

def save_to_pickle(glitch_tokens, glitch_token_ids, save_dir="glitch_tokens.pkl"):
    saved_object = {
        "glitch_tokens": glitch_tokens,
        "glitch_token_ids": glitch_token_ids
    }

    with open(save_dir, "wb") as f:
        pickle.dump(saved_object, f)

def save_to_json(all_entropy_data, save_dir="glitch_tokens.json"):
    try:
        with open(save_dir, 'w', encoding='utf-8') as f:
            json.dump(all_entropy_data, f, ensure_ascii=False, indent=4)
        print(f"✅ Entropy data successfully saved to {save_dir}")
    except Exception as e:
        print(f"⚠️ An error occurred while saving the file: {e}")


def main(args):
    print(f"Setting up model: {args.model_id}")
    model, tokenizer = setup_model_and_tokenizer(args.model_id, args.device)
    print(f"Model loaded on device: {model.device}")

    if args.fast:
        print("Running GlitchMiner (fast mode)...")
        glitch_tokens, glitch_token_ids = get_glitch_tokens(
            model,
            tokenizer,
            num_iterations=args.num_iterations,
            batch_size=args.batch_size,
            k=args.k,
            if_print=args.if_print
        )

        save_to_pickle(glitch_tokens, glitch_token_ids, save_dir=args.save_dir)
        print(f"Glitch tokens and their IDs have been saved to {args.save_dir}")
    else:
        print("Calculating entropy for all tokens (comprehensive mode)...")
        all_entropy_data = get_all_entropy_data(
            model,
            tokenizer,
            chat_template=args.chat_template
        )
        save_to_json(all_entropy_data, args.output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glitch Token Generator")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID to use")
    parser.add_argument("--device", type=str, default=None, help="Device to run the model on, e.g., 'cuda:0' or 'cpu'. Defaults to 'auto'.")
    parser.add_argument("--fast", action="store_true", help="Run fast glitch token mining instead of comprehensive entropy calculation.")
    
    # --- Arguments for fast mode ---
    parser.add_argument("--num_iterations", type=int, default=125, help="Number of iterations (fast mode)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (fast mode)")
    parser.add_argument("--k", type=int, default=32, help="Top k tokens to consider (fast mode)")
    parser.add_argument("--if_print", type=bool, default=True, help="Whether to print progress (fast mode)")
    parser.add_argument("--save_dir", type=str, default="glitch_tokens.pkl", help="Output file for fast mode (.pkl)")

    # --- Arguments for comprehensive mode ---
    parser.add_argument("--output_file_path", type=str, default="token_entropy.json", help="Output file for comprehensive mode (.json)")
    parser.add_argument("--chat_template", type=str, default=None, help="Chat template to use for input formatting (comprehensive mode)")

    args = parser.parse_args()
    main(args)