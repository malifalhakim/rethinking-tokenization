import json
from typing import List

class TokenNorm:
    def __init__(self, file_path:str, tokenizer):
        self.magikarp = self.load_magikarp_data(file_path)
        self.tokenizer = tokenizer

    def load_magikarp_data(self, filepath:str) -> dict:
        """
        Loads the Magikarp JSONL file, storing the 'main_indicator' for each token.
        """
        magikarp_data = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        token = data.get('i')
                        if token:
                            magikarp_data[token] = {
                                'parameter': data.get('main_indicator', None),
                            }
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from line: {line.strip()}")
        except FileNotFoundError:
            print(f"Error: Magikarp file not found at {filepath}")
            return None
        return magikarp_data
    
    def get_score(self, tokenization:List[str]) -> float:
        """
        Calculate the average 'main_indicator' for the given tokenization.
        Tokens not found in Magikarp data are ignored.
        """
        if not tokenization:
            return 0.0
        
        ids_tokenization = self.tokenizer.convert_tokens_to_ids(tokenization)
        norms = []
        num_counted = 0
        for id_token in ids_tokenization:
            norm = self.magikarp.get(id_token, {}).get('parameter', None)
            if not norm:
                continue
            norms.append(norm)
            num_counted += 1

        if num_counted == 0:
            return 0.0

        return sum(norms) / num_counted