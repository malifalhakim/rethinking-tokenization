import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import random
import re

from tokenizer.bpe_random_tokenizer import BPEAlternativeTokenizer
from typing import List, Set, Optional

def is_random_bpe(
        alternative_tokenizations: List[str], 
        canonical_tokenization: List[str]
    ) -> bool:
    """
    Efficiently checks if any token in canonical_tokenization is formed by 
    concatenating two or more adjacent tokens from alternative_tokenizations.
    """
    if len(alternative_tokenizations) < 2:
        return False

    possible_concatenations: Set[str] = set()
    n = len(alternative_tokenizations)
    for i in range(n - 1):
        current_concat = alternative_tokenizations[i]
        for j in range(i + 1, n):
            current_concat += alternative_tokenizations[j]
            possible_concatenations.add(current_concat)

    return any(token in possible_concatenations for token in canonical_tokenization)

class BPEAlternativeTokenizerFiltered(BPEAlternativeTokenizer):
    """
    A wrapper for BPE-based Hugging Face tokenizers to generate
    alternative tokenizations, and filter so there is no Random BPE tokenizations.
    """
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.protected_tokens = set(getattr(tokenizer, "all_special_tokens", []))
        self._protected_like_pattern = ("<|", "|>")

    def is_random_bpe(
        self,
        alternative_tokenizations: List[str], 
        canonical_tokenization: List[str]
    ) -> bool:
        """
        Efficiently checks if any token in canonical_tokenization is formed by 
        concatenating two or more adjacent tokens from alternative_tokenizations.
        """
        return is_random_bpe(alternative_tokenizations, canonical_tokenization)

    def _generate_one_word_tokenization(self, word: str) -> Optional[List[str]]:
        """
        Generates a single random tokenization for a word from its lattice.
        Returns None if the word cannot be tokenized.
        """
        if word in self.protected_tokens or (word.startswith("<|") and word.endswith("|>")):
            return [word]

        nodes = [[] for _ in range(len(word) + 1)]
        nodes[0].append(0)
        for i in range(1, len(word) + 1):
            for j in range(i):
                substring = word[j:i]
                if nodes[j] and substring in self.vocab:
                    nodes[i].append(j)

        if not nodes[len(word)]:
            return None

        word_tokens = []
        end_pos = len(word)
        while end_pos > 0:
            possible_starts = nodes[end_pos]
            start_pos = random.choice(possible_starts)
            token = word[start_pos:end_pos]
            word_tokens.insert(0, token)
            end_pos = start_pos
        
        return word_tokens

    def generate_alternatives(self, text: str, n: int) -> List[List[str]]:
        """
        Generates up to n alternative tokenizations that are NOT random BPE
        segmentations of the canonical tokenization by validating at the word level.
        """
        pre_words = self._get_pre_tokenized_words(text)

        canonical_tokens = tuple(self.tokenizer.tokenize(text))
        generated_tokenizations = {canonical_tokens}
        alternatives = []
        
        max_attempts = n * 10 + 50 
        for _ in range(max_attempts):
            if len(alternatives) >= n:
                break

            current_full_tokenization = []

            for word, offset in pre_words:
                begin, end = offset
                span_text = text[begin:end]
                canonical_word_tokens = self.tokenizer.tokenize(span_text)

                # If this span is already (exactly) a single protected token, keep it
                if (
                    (len(canonical_word_tokens) == 1 and canonical_word_tokens[0] in self.protected_tokens)
                    or span_text in self.protected_tokens
                    or (span_text.startswith("<|") and span_text.endswith("|>"))
                ):
                    current_full_tokenization.extend(canonical_word_tokens)
                    continue

                word_level_max_attempts = 10
                found_alternative_for_word = False
                for _ in range(word_level_max_attempts):
                    word_tokens = self._generate_one_word_tokenization(word)

                    if word_tokens is None:
                        current_full_tokenization.append(word)
                        found_alternative_for_word = True
                        break

                    if not self.is_random_bpe(word_tokens, canonical_word_tokens):
                        current_full_tokenization.extend(word_tokens)
                        found_alternative_for_word = True
                        break 
                
                if not found_alternative_for_word:
                    current_full_tokenization.extend(canonical_word_tokens)

            tokenization_tuple = tuple(current_full_tokenization)
            if tokenization_tuple and tokenization_tuple not in generated_tokenizations:
                generated_tokenizations.add(tokenization_tuple)
                alternatives.append(list(tokenization_tuple))
        
        if not alternatives:
            alternatives.append(list(canonical_tokens))

        return alternatives

    def _get_pre_tokenized_words(self, text: str):
        """
        Extends parent pre-tokenization by merging chat special tokens
        that were split into pieces like: <| , im , _start , |> -> <|im_start|>
        """
        base = super()._get_pre_tokenized_words(text)
        merged = []
        i = 0
        while i < len(base):
            token, (s, e) = base[i]

            if token == "<|" and i + 1 < len(base):
                # --- search for closing '|>' ---
                closing_idx = None
                for j in range(i + 1, len(base)):
                    if base[j][0] == "|>":
                        closing_idx = j
                        break

                if closing_idx is not None:
                    full_start = s
                    full_end = base[closing_idx][1][1]
                    candidate = text[full_start:full_end]
                    # --- Accept if ---:
                    # 1) in protected token already, OR
                    # 2) matches generic chat special pattern <|...|>
                    if (
                        candidate in self.protected_tokens
                        or re.fullmatch(r"<\|[^\s]{1,100}\|>", candidate)
                    ):
                        merged.append((candidate, (full_start, full_end)))
                        i = closing_idx + 1
                        continue

            # --- default: keep token ---
            merged.append((token, (s, e)))
            i += 1
        return merged
