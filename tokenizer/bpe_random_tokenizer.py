import re
import random
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import List, Set, Tuple

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

    def _get_pre_tokenized_words(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Normalizes and pre-tokenizes the text into words with their original
        character offsets, while correctly handling and preserving special tokens.
        """
        if not isinstance(self.tokenizer, PreTrainedTokenizerFast):
            print("WARNING: Using a non-fast tokenizer may lead to suboptimal results.")
            raise NotImplementedError("Non-fast tokenizers are not supported.")

        special_tokens = set()
        
        if self.tokenizer.all_special_tokens:
            special_tokens.update(self.tokenizer.all_special_tokens)
        
        vocab = self.tokenizer.get_vocab()
        for token in vocab.keys():
            if (token.startswith('<') and token.endswith('>')) or \
               (token.startswith('[') and token.endswith(']')) or \
               (token.startswith('<|') and token.endswith('|>')):
                special_tokens.add(token)
        
        special_tokens = list(special_tokens)
        
        if not special_tokens:
            return self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)

        escaped_tokens = [re.escape(token) for token in special_tokens]
        pattern = f"({ '|'.join(escaped_tokens) })"

        final_tuples = []
        last_end = 0

        for match in re.finditer(pattern, text):
            start, end = match.span()
            if start > last_end:
                normal_text_chunk = text[last_end:start]
                pre_tokenized_tuples = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(normal_text_chunk)
                
                for word, (local_start, local_end) in pre_tokenized_tuples:
                    final_tuples.append((word, (last_end + local_start, last_end + local_end)))

            special_token = match.group(0)
            final_tuples.append((special_token, (start, end)))
            
            last_end = end

        if last_end < len(text):
            remaining_chunk = text[last_end:]
            pre_tokenized_tuples = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(remaining_chunk)

            for word, (local_start, local_end) in pre_tokenized_tuples:
                final_tuples.append((word, (last_end + local_start, last_end + local_end)))

        return final_tuples

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
            
            for word, _ in pre_words:
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