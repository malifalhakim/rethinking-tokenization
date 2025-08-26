from tokenization_scorer import score
from typing import List

def renyi_score(tokenization:List[str], alpha:float=2.5) -> float:
    """Calculates the Renyi score of a given tokenization"""
    return score(tokenization, metric='renyi', power=alpha)