import sys
import os
import psutil
from typing import List, Optional, Dict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tokenizer.bpe_random_tokenizer_filtered import BPEAlternativeTokenizerFiltered
from quantifier.trainness.magikarp import TokenNorm

class BPENormTokenizer(BPEAlternativeTokenizerFiltered):
    """
    A wrapper for BPE-based tokenizers that generates a single tokenization 
    that aims to maximize the average token norm, while also filtering out
    "random BPE" segmentations.
    """

    def __init__(self, tokenizer, token_norm: TokenNorm, max_cache_size: int = 1000):
        """
        Initializes the norm-based tokenizer.
        """
        super().__init__(tokenizer)
        self.token_norm = token_norm
        self._memo_all_tokenizations: Dict[str, Optional[List[List[str]]]] = {}
        self.max_cache_size = max_cache_size
        self.process = psutil.Process(os.getpid())
    
    def _check_memory_safe(self) -> bool:
        """Quick memory check to avoid OOM."""
        try:
            available = psutil.virtual_memory().available
            total_memory = psutil.virtual_memory().total
            return available > (total_memory * 0.1)
        except:
            return True

    def _generate_all_word_tokenizations(self, word: str, canonical_tokenization: List[str]) -> Optional[List[List[str]]]:
        """
        Generates all possible valid tokenizations for a given word by traversing
        its tokenization lattice.
        """
        if len(word) > 25 or not self._check_memory_safe():
            self._memo_all_tokenizations.clear()
            return None

        if len(self._memo_all_tokenizations) > self.max_cache_size:
            self._memo_all_tokenizations.clear()

        if word in self._memo_all_tokenizations:
            return self._memo_all_tokenizations[word]

        nodes = [[] for _ in range(len(word) + 1)]
        for i in range(1, len(word) + 1):
            for j in range(i):
                substring = word[j:i]
                if substring in self.vocab:
                    if j == 0 or nodes[j]:
                        nodes[i].append(j)

        if not nodes[len(word)]:
            self._memo_all_tokenizations[word] = None
            return None

        memo_paths = {}
        def find_paths_recursive(end_pos):
            if not self._check_memory_safe():
                raise MemoryError("Memory limit reached during tokenization generation.")

            if end_pos in memo_paths:
                return memo_paths[end_pos]
            if end_pos == 0:
                return [[]]

            all_paths = []
            for start_pos in nodes[end_pos]:
                token = word[start_pos:end_pos]
                sub_paths = find_paths_recursive(start_pos)
                for path in sub_paths:
                    candidate_path = path + [token]
                    if not self.is_random_bpe(candidate_path, canonical_tokenization):
                        all_paths.append(candidate_path)
            
            memo_paths[end_pos] = all_paths
            return all_paths

        try:
            result = find_paths_recursive(len(word))
            self._memo_all_tokenizations[word] = result
            return result
        except MemoryError:
            self._memo_all_tokenizations.clear()
            self._memo_all_tokenizations[word] = None
            return None

    def _find_best_word_tokenization(self, word: str, canonical_tokenization: List[str]) -> List[str]:
        """
        Finds the tokenization for a single word that maximizes token norm and is
        not a "random BPE" segmentation.
        """
        all_tokenizations = self._generate_all_word_tokenizations(word, canonical_tokenization)
        
        if not all_tokenizations:
            return canonical_tokenization

        best_tokenization = canonical_tokenization
        max_score = self.token_norm.get_score(canonical_tokenization)

        for tokenization in all_tokenizations:
            if tokenization == canonical_tokenization:
                continue
            
            score = self.token_norm.get_score(tokenization)
            if score > max_score:
                max_score = score
                best_tokenization = tokenization
        
        return best_tokenization

    def generate_best_tokenization(self, text: str) -> List[str]:
        """
        Generates a single tokenization for the entire text that aims to
        maximize the overall average token norm by optimizing word by word.
        """
        self._memo_all_tokenizations.clear()
        
        pre_words = self._get_pre_tokenized_words(text)
        final_tokenization = []
        
        for word, offset in pre_words:
            begin, end = offset
            canonical_word_tokens = self.tokenizer.tokenize(text[begin:end])
            
            best_word_tokens = self._find_best_word_tokenization(word, canonical_word_tokens)
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