import re
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
                                'type': data.get('magikarp', None),
                                'raw_vocab': data.get('raw_vocab', None),
                                'decoded': data.get('decoded', None)
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
    
    def is_contains_undertrained_tokens(self, segment:str, threshold:str='strong_verified') -> bool:
        """
        Check if any token in the segment has undertrained tokens above the given threshold.
        Thresholds: 'weak_verified', 'strong_verified'
        """
        if not segment:
            return False
        
        segment_tokens = self.tokenizer(segment, add_special_tokens=False).tokens()
        ids_tokenization = self.tokenizer.convert_tokens_to_ids(segment_tokens)
        for id_token in ids_tokenization:
            type_token = self.magikarp.get(id_token, {}).get('type', None)
            if threshold == 'weak_verified' and type_token in ['weak_verified', 'strong_verified']:
                return True
            if threshold == 'strong_verified' and type_token == 'strong_verified':
                return True
            
        return False

    def get_selected_undertrained_tokens(self, threshold: str = 'strong_verified') -> List[str]:
        """
        Retrieve all tokens that are not gibberish and are marked as undertrained.
        """
        selected_tokens = {}
        for token_id, data in self.magikarp.items():
            type_token = data.get('type', None)

            if threshold == 'weak_verified' and type_token not in ['weak_verified', 'strong_verified']:
                continue
            if threshold == 'strong_verified' and type_token != 'strong_verified':
                continue

            parameter = data.get('parameter', None)
            decoded_token = data.get('decoded', '')
            raw_vocab = data.get('raw_vocab', '')
            token_str = decoded_token

            if not isinstance(token_str, str):
                continue

            if not token_str or token_str.strip() == '':
                continue

            if not token_str.isprintable():
                continue

            if not re.search(r'[a-zA-Z]', token_str):
                continue

            if len(token_str.strip()) < 4:
                continue

            if r"\ufffd" in token_str:
                continue

            if re.search(r'[\{\}\[\]\(\)]', token_str):
                continue

            if (token_str.startswith('<') and token_str.endswith('>')) or \
               (token_str.startswith('[') and token_str.endswith(']')) or \
               (token_str.startswith('<|') and token_str.endswith('|>')):
                continue

            selected_tokens[token_id] = {
                'raw_vocab': raw_vocab,
                'decoded': decoded_token,
                'parameter': parameter,
                'type': type_token
            }
        
        sorted_tokens = dict(sorted(
            selected_tokens.items(),
            key=lambda x: len(x[1].get('decoded', '')),
            reverse=True
        ))
        
        return sorted_tokens