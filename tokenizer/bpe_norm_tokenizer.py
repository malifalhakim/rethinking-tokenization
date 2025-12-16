import math
import heapq
import sys
import os
from typing import List, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tokenizer.bpe_random_tokenizer import BPEAlternativeTokenizer
from quantifier.trainness.magikarp import TokenNorm

class BPENormTokenizer(BPEAlternativeTokenizer):
    """
    A wrapper for BPE-based tokenizers that generates a single tokenization 
    that aims to maximize the average token norm
    """

    def __init__(self, tokenizer, token_norm: TokenNorm):
        """
        Initializes the norm-based tokenizer.
        """
        super().__init__(tokenizer)
        self.token_norm = token_norm
    
    def _scoring_function(self, item: Tuple[float, int, List[str]], length_penalty: float = 0.02) -> float:
        """
        Scoring function to evaluate tokenizations based on average token norm.
        """
        total_score, token_count, _ = item
        if token_count == 0:
            return -math.inf
        
        average_score = total_score / token_count
        penalty_term = length_penalty * (token_count - 1)
        return average_score - penalty_term

    def _find_best_word_tokenization(self, word: str, k: int = 5) -> List[str]:
        """
        Finds the optimal tokenization for a word based on token norm scores.
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
                    token_score = self.token_norm.get_score([substring])

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
        
        if dp[n]:
            best_candidate = max(dp[n], key=lambda x: self._scoring_function(x))
            return best_candidate[2]
        
        return self.tokenizer.tokenize(word)

    def generate_best_tokenization(self, text: str) -> List[str]:
        """
        Generates a single tokenization for the entire text that aims to
        maximize the overall average token norm by optimizing word by word.
        """
        pre_words = self._get_pre_tokenized_words(text)
        final_tokenization = []
        
        for word, offset in pre_words:
            best_word_tokens = self._find_best_word_tokenization(word)
            final_tokenization.extend(best_word_tokens)
            
        return final_tokenization

    def generate_alternatives(self, text: str, n: int = 1) -> List[List[str]]:
        """
        Overrides the parent method. Instead of generating random alternatives,
        it returns a list containing only the single best tokenization based on norm.
        The 'n' parameter is ignored.
        """
        best_tokenization = self.generate_best_tokenization(text)
        return [best_tokenization]