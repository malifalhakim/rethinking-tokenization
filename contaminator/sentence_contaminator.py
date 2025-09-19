import json
import pickle
import random
from typing import List, Optional, Union
from pathlib import Path


class Contaminator:
    """
    A class to contaminate sentences with noise using undertrained tokens.
    """
    
    PREFIXES = [
        "un", "re", "in", "ex",
        "bi", "co", "de", "en"
    ]

    SUFFIXES = [
        "ed", "er", "es", "en",
        "ly", "al", "ic", "or"
    ]
    
    def __init__(self, tokenizer, magikarp_path: Optional[Union[str, Path]] = None, 
                 glitchminer_path: Optional[Union[str, Path]] = None):
        """
        Initialize the Contaminator with a tokenizer and path to undertrained tokens.
        
        Args:
            tokenizer: The tokenizer to use for token operations
            magikarp_path: Path to magikarp output file (JSONL format)
            glitchminer_path: Path to glitchminer output file (pickle format)
        """
        self.tokenizer = tokenizer
        self.undertrained_tokens = self._load_undertrained_tokens(
            magikarp_path, glitchminer_path
        )

    def _load_undertrained_tokens(self, magikarp_path: Optional[Union[str, Path]], 
                                 glitchminer_path: Optional[Union[str, Path]]) -> List[str]:
        """Load undertrained tokens from the provided file path."""
        if magikarp_path:
            return self._load_magikarp_tokens(magikarp_path)
        elif glitchminer_path:
            return self._load_glitchminer_tokens(glitchminer_path)
        else:
            raise ValueError("Please provide either magikarp_path or glitchminer_path.")

    def _load_magikarp_tokens(self, file_path: Union[str, Path]) -> List[str]:
        """Extract undertrained tokens from Magikarp output file."""
        tokens = []
        file_path = Path(file_path)
        
        try:
            with file_path.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if data.get('magikarp') == 'strong_verified' and data.get('category') == 'OK':
                            if decoded := data.get('decoded'):
                                tokens.append(decoded)
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON at line {line_num}: {line.strip()}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Magikarp file not found: {file_path}")
        
        if not tokens:
            print("Warning: No valid undertrained tokens found in magikarp file")
        
        return tokens
    
    def _load_glitchminer_tokens(self, file_path: Union[str, Path]) -> List[str]:
        """Extract undertrained tokens from GlitchMiner output pickle file."""
        file_path = Path(file_path)
        
        try:
            with file_path.open('rb') as f:
                data = pickle.load(f)
            
            tokens = data.get('glitch_tokens', [])
            if not tokens:
                print("Warning: No glitch_tokens found in glitchminer file")
            
            return tokens
        except FileNotFoundError:
            raise FileNotFoundError(f"GlitchMiner file not found: {file_path}")
        except (pickle.UnpicklingError, KeyError) as e:
            raise ValueError(f"Invalid GlitchMiner file format: {e}")
    
    def _create_noise_word(self, base_token: str) -> str:
        """Create a noise word by adding prefix or suffix to base token."""
        if base_token.startswith(' '):
            suffix = random.choice(self.SUFFIXES)
            return base_token + suffix
        else:
            prefix = random.choice(self.PREFIXES)
            return prefix + base_token
        
    def _contains_undertrained_token(self, text: str) -> bool:
        """Check if text contains any undertrained token after tokenization."""
        tokens = self.tokenizer.tokenize(text)
        undertrained_set = set(self.undertrained_tokens) 
        return any(token in undertrained_set for token in tokens)

    def get_noise_word(self) -> str:
        """
        Generate a noise word containing undertrained tokens.
            
        Returns:
            A noise word containing undertrained tokens
            
        Raises:
            ValueError: If no undertrained tokens available or max attempts exceeded
        """
        if not self.undertrained_tokens:
            raise ValueError("No undertrained tokens available.")
        
        while True:
            base_token = random.choice(self.undertrained_tokens)
            noise_word = self._create_noise_word(base_token)
            
            if self._contains_undertrained_token(noise_word):
                return noise_word

    def contaminate_sentence(self, sentence: str) -> str:
        """
        Contaminate sentence by adding noise word at the beginning.
        
        Args:
            sentence: The sentence to contaminate
            
        Returns:
            Contaminated sentence with noise word prefix
        """
        if not sentence.strip():
            raise ValueError("Input sentence cannot be empty.")

        noise_word = self.get_noise_word()
        return f"{noise_word}, {sentence}"