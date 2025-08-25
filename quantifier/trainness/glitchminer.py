import argparse
import pickle

from transformers import AutoTokenizer, AutoModelForCausalLM
from glitchminer import GlitchMiner

def get_glitch_tokens(model_id:str, num_iterations:int=125, batch_size:int=8, k:int=32, if_print:bool=True):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

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

def save_to_pickle(glitch_tokens, glitch_token_ids, save_dir="glitch_tokens.pkl"):
    saved_object = {
        "glitch_tokens": glitch_tokens,
        "glitch_token_ids": glitch_token_ids
    }

    with open(save_dir, "wb") as f:
        pickle.dump(saved_object, f)

def main(args):
    glitch_tokens, glitch_token_ids = get_glitch_tokens(
        model_id=args.model_id,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        k=args.k,
        if_print=args.if_print
    )

    save_to_pickle(glitch_tokens, glitch_token_ids, save_dir=args.save_dir)
    print(f"Glitch tokens and their IDs have been saved to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glitch Token Generator")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID to use")
    parser.add_argument("--num_iterations", type=int, default=125, help="Number of iterations")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--k", type=int, default=32, help="Top k tokens to consider")
    parser.add_argument("--if_print", type=bool, default=True, help="Whether to print progress")
    parser.add_argument("--save_dir", type=str, default="glitch_tokens.pkl", help="Directory to save the glitch tokens")

    args = parser.parse_args()
    main(args)
