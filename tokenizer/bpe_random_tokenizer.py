import re
import random
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import List, Set, Tuple, Union, Optional, Dict, Any

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
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token = tokenizer.eos_token
        self.vocab: Set[str] = set(tokenizer.get_vocab().keys())
        self.padding_side = tokenizer.padding_side

    def __call__(
        self,
        text: Union[str, List[str]],
        n: int = 1,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        return_tensors: Optional[str] = 'pt',
    ) -> Dict[str, Any]:
        """
        Tokenizes the input text(s), generating N alternative tokenizations.

        Args:
            text: A single string or a list of strings to tokenize.
            n: Number of alternative tokenizations to generate per input.
            add_special_tokens: Whether to add special tokens.
            padding: Whether to pad the sequences. Can be True, False, or 'longest'.
            return_tensors: The type of tensors to return ('pt' for PyTorch).
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        batch_input_ids = []
        for t in texts:
            input_ids = self.encode(t, n=n, return_tensors=return_tensors,add_special_tokens=add_special_tokens)
            batch_input_ids.append(input_ids)
        
        flat_input_ids = []
        for ids_list in batch_input_ids:
            flat_input_ids.extend(ids_list)
        
        if padding:
            max_length = max(ids.shape[1] for ids in batch_input_ids)
            padded_input_ids = []
            attention_masks = []

            for ids in flat_input_ids:
                seq_length = ids.shape[1]
                pad_length = max_length - seq_length

                if pad_length > 0:
                    pad_id = self.tokenizer.pad_token_id
                    padding_tensor = torch.full((ids.shape[0], pad_length), pad_id, dtype=ids.dtype)

                    if self.padding_side == "left":
                        ids = torch.cat([padding_tensor, ids], dim=1)
                        attention_mask = torch.cat([
                            torch.zeros(ids.shape[0], pad_length, dtype=torch.long),
                            torch.ones(ids.shape[0], seq_length, dtype=torch.long)
                        ], dim=1)
                    else:
                        ids = torch.cat([ids, padding_tensor], dim=1)
                        attention_mask = torch.cat([
                            torch.ones(ids.shape[0], seq_length, dtype=torch.long),
                            torch.zeros(ids.shape[0], pad_length, dtype=torch.long)
                        ], dim=1)
                else:
                    attention_mask = torch.ones(ids.shape[0], seq_length, dtype=torch.long)
                
                attention_masks.append(attention_mask)
                padded_input_ids.append(ids)

            return {
                "input_ids": torch.cat(padded_input_ids, dim=0),
                "attention_mask": torch.cat(attention_masks, dim=0)
            }
        
        return {
            "input_ids": torch.cat(batch_input_ids, dim=0)
        }
    
    def apply_chat_template(self, chat: List[Dict[str, str]], tokenize: bool = True, add_generation_prompt: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Applies a chat template to the conversation.

        Args:
            chat: A list of dictionaries representing the chat messages.
            tokenize: Whether to tokenize the formatted chat.
            add_generation_prompt: Whether to add a generation prompt at the end.
        """
        formatted = self.tokenizer.apply_chat_template(
            chat, 
            tokenize=False, 
            add_generation_prompt=add_generation_prompt
        )

        if tokenize:
            return self(text=formatted)
        else:
            return formatted

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

    def encode(self, text:str, n:int=1, return_tensors:str='pt', add_special_tokens:bool=True):
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
    
    def convert_ids_to_tokens(self, ids) -> List[str]:
        """
        Converts a list of token IDs back to tokens.
        """
        return self.tokenizer.convert_ids_to_tokens(ids)

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back to a string.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)