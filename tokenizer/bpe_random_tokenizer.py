import random
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import List, Set

class BPEAlternativeTokenizer:
    """
    A wrapper for BPE-based Hugging Face tokenizers to generate
    alternative tokenizations.

    This implementation follows the specific workflow for BPE:
    1. Normalize the text.
    2. Pre-tokenize the text.
    3. For each word, build a subword lattice using a fast vocabulary lookup.
    4. Generate alternatives by sampling paths from the lattices.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initializes the BPEAlternativeTokenizer.
        """
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            print(
                "Warning: For best results and correctness with BPE, please use a "
                "'fast' tokenizer."
            )
        self.tokenizer = tokenizer
        self.vocab: Set[str] = set(tokenizer.get_vocab().keys())

    def _get_pre_tokenized_words(self, text: str) -> List[str]:
        """
        Normalizes and then pre-tokenizes the text into words.
        """
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            normalized_text = self.tokenizer.backend_tokenizer.normalizer.normalize_str(text)
            return [word for word, _ in self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(normalized_text)]
        else:
            print("WARNING: Using a non-fast tokenizer may lead to suboptimal results.")
            return self.tokenizer.tokenize(text)

    def generate_alternatives(self, text: str, n: int) -> List[List[str]]:
        """
        Generates N different, non-canonical tokenizations for the given text.
        """
        pre_words = self._get_pre_tokenized_words(text)
        
        canonical_tokens = tuple(self.tokenizer.tokenize(text))
        generated_tokenizations = {canonical_tokens}
        alternatives = []

        max_attempts = n * 10 + 20
        for _ in range(max_attempts):
            if len(alternatives) >= n:
                break

            current_full_tokenization = []
            
            for word in pre_words:
                nodes = [[] for _ in range(len(word) + 1)]
                nodes[0].append(0)
                for i in range(1, len(word) + 1):
                    for j in range(i):
                        substring = word[j:i]
                        if nodes[j] and substring in self.vocab:
                            nodes[i].append(j)

                if not nodes[len(word)]:
                    current_full_tokenization.append(word)
                    continue

                word_tokens = []
                end_pos = len(word)
                while end_pos > 0:
                    possible_starts = nodes[end_pos]
                    start_pos = random.choice(possible_starts)
                    token = word[start_pos:end_pos]
                    word_tokens.insert(0, token)
                    end_pos = start_pos
                
                current_full_tokenization.extend(word_tokens)

            if current_full_tokenization:
                tokenization_tuple = tuple(current_full_tokenization)
                if tokenization_tuple not in generated_tokenizations:
                    generated_tokenizations.add(tokenization_tuple)
                    alternatives.append(list(tokenization_tuple))

        return alternatives

    def encode(self, text:str, n:int=10, return_tensors:str='pt', add_special_tokens:bool=True):
        """
        Encodes the text into token IDs, generating N alternative tokenizations.
        """
        alternatives = self.generate_alternatives(text, n)
        encoded_tensors = []
        
        for alt in alternatives:
            ids = self.tokenizer.convert_tokens_to_ids(alt)
            if add_special_tokens:
                ids = self.tokenizer.build_inputs_with_special_tokens(ids)
            if return_tensors == 'pt':
                ids = torch.tensor([ids])
            encoded_tensors.append(ids)

        return encoded_tensors