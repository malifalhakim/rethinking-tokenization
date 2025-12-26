import sys
import os
import psutil
from typing import List, Optional, Dict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tokenizer.bpe_entropy_tokenizer import BPEEntropyTokenizer
from quantifier.trainness.entropy import TokenEntropy

class BPEUndertrainedEntropyTokenizer(BPEEntropyTokenizer):
    """
    A wrapper for BPE-based tokenizers that generates a single tokenization 
    that aims to maximize the average token entropy, specifically focusing on words
    containing undertrained tokens.
    """

    def __init__(self, tokenizer, token_entropy: TokenEntropy):
        """
        Initializes the entropy-based tokenizer.
        """
        super().__init__(tokenizer, token_entropy)

    def generate_best_tokenization(self, text: str) -> List[str]:
        """
        Generates a single tokenization for the entire text that aims to
        maximize the overall average token entropy by optimizing word by word. Only
        words containing undertrained tokens are re-tokenized; others use the canonical
        tokenization.
        """
        pre_words = self._get_pre_tokenized_words(text)
        final_tokenization = []
        
        for word, offset in pre_words:
            if word in self.special_tokens:
                final_tokenization.append(word)
                continue

            begin, end = offset
            canonical_word_tokens = self.tokenizer.tokenize(text[begin:end])
            if not self.token_entropy.is_contains_undertrained_tokens(text[begin:end]):
                final_tokenization.extend(canonical_word_tokens)
                continue
            
            original_word = text[begin:end]
            best_word_tokens = self._find_best_word_tokenization(word, original_word)
            final_tokenization.extend(best_word_tokens)
            
        return final_tokenization