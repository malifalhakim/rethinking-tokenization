from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable, Sequence, Tuple
import difflib

import argparse
import json

__all__ = [
    "TokenDiffOp",
    "diff_tokens",
    "highlight_ansi",
    "highlight_markdown",
    "highlight_html",
    "side_by_side",
]

@dataclass
class TokenDiffOp:
    """
    Represents one diff operation between two token lists.
    tag: one of 'equal', 'replace', 'delete', 'insert'
    a_span / b_span: (start, end) indices in the original sequences
    a_tokens / b_tokens: the actual token slices involved
    """
    tag: str
    a_span: Tuple[int, int]
    b_span: Tuple[int, int]
    a_tokens: List[str]
    b_tokens: List[str]

def diff_tokens(a: Sequence[str], b: Sequence[str]) -> List[TokenDiffOp]:
    """
    Return a list of TokenDiffOp describing how to transform token list a into b.
    """
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ops: List[TokenDiffOp] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ops.append(
            TokenDiffOp(
                tag=tag,
                a_span=(i1, i2),
                b_span=(j1, j2),
                a_tokens=list(a[i1:i2]),
                b_tokens=list(b[j1:j2]),
            )
        )
    return ops

ANSI = {
    "reset": "\x1b[0m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "bright_black": "\x1b[90m",
    # Added colors for labels
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
}

def _format_join(tokens: Iterable[str], sep: str = " ") -> str:
    return sep.join(tokens)

def highlight_ansi(a: Sequence[str], b: Sequence[str], sep: str = " ") -> Tuple[str, str]:
    """
    Return (highlighted_a, highlighted_b) with ANSI color highlighting.
    delete: red (only in a)
    insert: green (only in b)
    replace: yellow (orig in a) / green (new in b)
    equal: default
    """
    ops = diff_tokens(a, b)
    out_a: List[str] = []
    out_b: List[str] = []
    for op in ops:
        if op.tag == "equal":
            text_a = _format_join(op.a_tokens, sep)
            out_a.append(text_a)
            out_b.append(text_a)
        elif op.tag == "delete":
            out_a.append(f"{ANSI['red']}{_format_join(op.a_tokens, sep)}{ANSI['reset']}")
        elif op.tag == "insert":
            out_b.append(f"{ANSI['green']}{_format_join(op.b_tokens, sep)}{ANSI['reset']}")
        elif op.tag == "replace":
            out_a.append(f"{ANSI['yellow']}{_format_join(op.a_tokens, sep)}{ANSI['reset']}")
            out_b.append(f"{ANSI['green']}{_format_join(op.b_tokens, sep)}{ANSI['reset']}")
    return (sep.join(out_a), sep.join(out_b))

def get_tokenizations_data(subject: str, question_index: int, raw_data: list) -> list:
    """Get tokenizations used for a specific subject & question_index."""
    tokenizations = []
    for detail in raw_data:
        if detail.get('question_index') == question_index and detail.get('subject') == subject:
            tokens_used = detail['tokens_used']
            tokenizations.append(
                {
                    "candidate_description": detail.get('candidate_description'),
                    "tokens_used": tokens_used,
                    "is_correct": detail.get('is_correct')
                }
            )
    return tokenizations

def read_json(file_path: str) -> dict:
    """Reads a JSON file and returns its content as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main(args):
    raw_data_1 = read_json(args.input_1)["per_candidate_results"]
    raw_data_2 = read_json(args.input_2)["per_candidate_results"]

    tokenizations_1 = get_tokenizations_data(
        subject=args.subject, question_index=args.question_index, raw_data=raw_data_1)
    tokenizations_2 = get_tokenizations_data(
        subject=args.subject, question_index=args.question_index, raw_data=raw_data_2)
    
    baseline_tokenization = tokenizations_1[0]
    for tokenization in tokenizations_2:
        if baseline_tokenization['is_correct'] != tokenization['is_correct']:
            ha, hb = highlight_ansi(baseline_tokenization['tokens_used'], tokenization['tokens_used'])
            baseline_correct = baseline_tokenization['is_correct']
            alt_correct = tokenization['is_correct']
            def fmt_bool(val: bool) -> str:
                return f"{ANSI['green']}{val}{ANSI['reset']}" if val else f"{ANSI['red']}{val}{ANSI['reset']}"
            print(f"{ANSI['cyan']}Baseline{ANSI['reset']}:\n {ha}")
            print(f"{ANSI['blue']}Is Correct{ANSI['reset']}: {fmt_bool(baseline_correct)}\n")
            print(f"{ANSI['magenta']}Alternative{ANSI['reset']}:\n {hb}")
            print(f"{ANSI['blue']}Is Correct{ANSI['reset']}: {fmt_bool(alt_correct)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze tokenization differences.")
    parser.add_argument("--input_1", type=str, required=True, help="Path to the first input JSON file.")
    parser.add_argument("--input_2", type=str, required=True, help="Path to the second input JSON file.")
    parser.add_argument("--subject", type=str, default="abstract_algebra", help="Subject to analyze.")
    parser.add_argument("--question_index", type=int, default=20, help="Question index to analyze.")
    args = parser.parse_args()

    main(args)