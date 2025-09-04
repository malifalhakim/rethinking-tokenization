from typing import List, Dict
import json

class TokenEntropy:
    def __init__(self, file_path: str, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer

        self.data = self._read_data()
        self.entropy_map = self._process_data(self.data)

    def _read_data(self) -> List[Dict]:
        """Read data from the JSON file and return a list of dicts."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Expected top-level JSON array.")
        return data

    def _process_data(self, raw_data: List[Dict]) -> Dict[str, float]:
        """Process the raw data into a more usable format."""
        return {item["token_id"]: (item["token"], item["entropy"]) for item in raw_data if "token_id" in item and "token" in item and "entropy" in item}

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