import math
import heapq
import sys
import os
from typing import List, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tokenizer.bpe_random_tokenizer import BPEAlternativeTokenizer
from quantifier.trainness.entropy import TokenEntropy

class BPEEntropyTokenizer(BPEAlternativeTokenizer):
    """
    A wrapper for BPE-based tokenizers that generates a single tokenization 
    that aims to maximize the average token entropy
    """

    def __init__(self, tokenizer, token_entropy: TokenEntropy):
        """
        Initializes the entropy-based tokenizer.
        """
        super().__init__(tokenizer)
        self.token_entropy = token_entropy
    
    def _scoring_function(self, item: Tuple[float, int, List[str]], length_penalty: float = 0.1) -> float:
        """
        Scoring function to evaluate tokenizations based on average token entropy.
        """
        total_score, token_count, _ = item
        if token_count == 0:
            return -math.inf
        
        average_score = total_score / token_count
        penalty_term = length_penalty * (token_count - 1)
        return average_score - penalty_term

    def _find_best_word_tokenization(self, word: str, original_word:str, k: int = 5) -> List[str]:
        """
        Finds the optimal tokenization for a word based on token entropy scores.
        """
        n = len(word)
        # Tuple format: (total_score, token_count, tokenization)
        dp: List[List[Tuple[float, int, List[str]]]] = [[] for _ in range(n + 1)]

        dp[0] = [(0.0, 0, [])]  # Base case: empty string

        for i in range(1, n + 1):
            candidates = []
            for j in range(i):
                substring = word[j:i]
                if substring in self.vocab:
                    token_score = self.token_entropy.get_score([substring])

                    if dp[j]:
                        for prev_total, prev_count, prev_tokens in dp[j]:
                            new_total = prev_total + token_score
                            new_count = prev_count + 1
                            new_tokens = prev_tokens + [substring]
                            candidates.append((new_total, new_count, new_tokens))
            
            if not candidates:
                continue

            # Keep top K candidates for this position 
            dp[i] = heapq.nlargest(k, candidates, key=lambda x: self._scoring_function(x))

        default_tokens = self.tokenizer.tokenize(original_word)
        default_score = sum([self.token_entropy.get_score([t]) for t in default_tokens])
        default_candidate = (default_score, len(default_tokens), default_tokens)
        
        if dp[n]:
            all_candidates = dp[n] + [default_candidate]
            best_candidate = max(all_candidates, key=lambda x: self._scoring_function(x))
            return best_candidate[2]
        
        return self.tokenizer.tokenize(original_word)

    def generate_best_tokenization(self, text: str) -> List[str]:
        """
        Generates a single tokenization for the entire text that aims to
        maximize the overall average token entropy by optimizing word by word.
        """
        pre_words = self._get_pre_tokenized_words(text)
        final_tokenization = []
        
        for word, offset in pre_words:
            if word in self.special_tokens:
                final_tokenization.append(word)
                continue

            begin, end = offset
            original_word = text[begin:end]
            best_word_tokens = self._find_best_word_tokenization(word, original_word)
            final_tokenization.extend(best_word_tokens)
            
        return final_tokenization

    def generate_alternatives(self, text: str, n: int = 1) -> List[List[str]]:
        """
        Overrides the parent method. Instead of generating random alternatives,
        it returns a list containing only the single best tokenization based on entropy.
        The 'n' parameter is ignored.
        """
        best_tokenization = self.generate_best_tokenization(text)
        return [best_tokenization]