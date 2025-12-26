from typing import List, Dict
import re
import json
import pickle

class TokenEntropy:
    def __init__(self, file_path: str, tokenizer, pkl_file_path: str = None):
        self.file_path = file_path
        self.pkl_file_path = pkl_file_path
        self.tokenizer = tokenizer

        self.data = self._read_data()
        self.entropy_map = self._process_data(self.data)
        if self.pkl_file_path:
            self.undertrained_tokens = self._read_pickle()
        else:
            self.undertrained_tokens = list()

    def _read_data(self) -> List[Dict]:
        """Read data from the JSON file and return a list of dicts."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Expected top-level JSON array.")
        return data
    
    def _read_pickle(self):
        """Read data from the pickle file."""
        with open(self.pkl_file_path, "rb") as f:
            data = pickle.load(f)
        return data['glitch_tokens']

    def _process_data(self, raw_data: List[Dict]) -> Dict[str, float]:
        """Process the raw data into a more usable format. Transform into a higher better score."""
        return {item["token_id"]: (item["token"], -1 * item["entropy"]) for item in raw_data if "token_id" in item and "token" in item and "entropy" in item}

    def get_score(self, tokenization: List[str]) -> float:
        """Average entropy of provided tokens (missing tokens count as 0)."""
        if not tokenization:
            return 0.0
        
        ids_tokenization = self.tokenizer.convert_tokens_to_ids(tokenization)
        entropys = []
        count = 0
        for token_id in ids_tokenization:
            entropy = self.entropy_map.get(token_id, (None, None))[1]
            if not entropy:
                continue
            entropys.append(entropy)
            count += 1
        return sum(entropys) / count if count > 0 else 0.0
    
    def is_contains_undertrained_tokens(self, segment: str) -> bool:
        """Check if any token in the segment is in the undertrained tokens list."""
        if not self.undertrained_tokens:
            print("ERROR: Please provide the pickle file path to load undertrained tokens.")
            return False

        if not segment:
            return False
        
        segment_tokens = self.tokenizer(segment, add_special_tokens=False).tokens()
        for token in segment_tokens:
            if token in self.undertrained_tokens:
                return True
            
        return False
    
    def get_selected_undertrained_tokens(self, threshold) -> List[str]:
        """Return the list of undertrained tokens. Threshold parameter is kept for compatibility with TokenNorm."""
        selected_tokens = {}
        for i, token_str in enumerate(self.undertrained_tokens):
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

            selected_tokens[i] = {
                'decoded': token_str
            }
        
        sorted_tokens = dict(sorted(
            selected_tokens.items(),
            key=lambda x: len(x[1].get('decoded', '')),
            reverse=True
        ))

        return sorted_tokens