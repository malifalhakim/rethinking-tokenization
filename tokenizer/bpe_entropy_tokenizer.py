import sys
import os
from typing import List, Optional, Dict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tokenizer.bpe_random_tokenizer_filtered import BPEAlternativeTokenizerFiltered
from quantifier.trainness.entropy import TokenEntropy

class BPEEntropyTokenizer(BPEAlternativeTokenizerFiltered):
    """
    A wrapper for BPE-based tokenizers that generates a single tokenization 
    that aims to minimize the average token entropy, while also filtering out
    "random BPE" segmentations.
    """

    def __init__(self, tokenizer, token_entropy: TokenEntropy):
        """
        Initializes the entropy-based tokenizer.
        """
        super().__init__(tokenizer)
        self.token_entropy = token_entropy
        self._memo_all_tokenizations: Dict[str, Optional[List[List[str]]]] = {}

    def _generate_all_word_tokenizations(self, word: str) -> Optional[List[List[str]]]:
        """
        Generates all possible valid tokenizations for a given word by traversing
        its tokenization lattice.
        """
        if word in self._memo_all_tokenizations:
            return self._memo_all_tokenizations[word]

        if word in self.protected_tokens or (word.startswith("<|") and word.endswith("|>")):
            self._memo_all_tokenizations[word] = [[word]]
            return [[word]]

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
            if end_pos in memo_paths:
                return memo_paths[end_pos]
            if end_pos == 0:
                return [[]]

            all_paths = []
            for start_pos in nodes[end_pos]:
                token = word[start_pos:end_pos]
                sub_paths = find_paths_recursive(start_pos)
                for path in sub_paths:
                    all_paths.append(path + [token])
            
            memo_paths[end_pos] = all_paths
            return all_paths

        result = find_paths_recursive(len(word))
        self._memo_all_tokenizations[word] = result
        return result

    def _find_best_word_tokenization(self, word: str, canonical_tokenization: List[str]) -> List[str]:
        """
        Chooses the tokenization with minimal entropy score (lower = better),
        excluding random-BPE segmentations. Protected tokens are kept as-is.
        """
        # If canonical already a single protected token, keep it
        if (
            (len(canonical_tokenization) == 1 and canonical_tokenization[0] in self.protected_tokens)
            or word in self.protected_tokens
            or (word.startswith("<|") and word.endswith("|>"))
        ):
            return canonical_tokenization

        all_tokenizations = self._generate_all_word_tokenizations(word)
        
        if not all_tokenizations:
            return canonical_tokenization

        best_tokenization = canonical_tokenization
        min_score = self.token_entropy.get_score(canonical_tokenization)

        for tokenization in all_tokenizations:
            if tokenization == canonical_tokenization:
                continue
            
            if self.is_random_bpe(tokenization, canonical_tokenization):
                continue
            
            score = self.token_entropy.get_score(tokenization)
            if score < min_score:
                min_score = score
                best_tokenization = tokenization
        
        return best_tokenization

    def generate_best_tokenization(self, text: str) -> List[str]:
        """
        Minimizes average token entropy span-by-span, preserving protected tokens.
        """
        self._memo_all_tokenizations.clear()
        
        pre_words = self._get_pre_tokenized_words(text)
        final_tokenization = []
        
        for word, offset in pre_words:
            begin, end = offset
            span_text = text[begin:end]
            canonical_word_tokens = self.tokenizer.tokenize(span_text)

            # Fast path for protected/special spans
            if (
                (len(canonical_word_tokens) == 1 and canonical_word_tokens[0] in self.protected_tokens)
                or span_text in self.protected_tokens
                or (span_text.startswith("<|") and span_text.endswith("|>"))
            ):
                final_tokenization.extend(canonical_word_tokens)
                continue

            best_word_tokens = self._find_best_word_tokenization(word, canonical_word_tokens)
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

