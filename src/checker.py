"""Answer checking with Hugging Face's math-verify library.

We use math-verify (https://github.com/huggingface/Math-Verify) rather than
writing our own symbolic checker. It parses LaTeX answers into SymPy and
checks mathematical equivalence, e.g. 1/2 == 0.5 == \\frac{1}{2}.

Functions:
    is_correct(completion, gold)   -> bool, the reward used for training (1 or 0)
    extract_answer(completion)     -> str or None, the model's final answer, used
                                      to group samples for the confident-error rate
    has_valid_format(completion)   -> bool, whether a parseable \\boxed{} answer exists
"""

from math_verify import parse, verify

from data import last_boxed_content


def _parse_gold(gold):
    # Wrapping the reference answer in $...$ tells math-verify it is LaTeX.
    return parse(f"${gold}$")


def is_correct(completion, gold):
    """True if the completion's final answer is equivalent to the gold answer."""
    try:
        pred = parse(completion)
        if not pred:
            return False
        return bool(verify(_parse_gold(gold), pred))
    except Exception:
        # math-verify can raise on unusual LaTeX; count that as incorrect.
        return False


def extract_answer(completion):
    """The model's final answer as a string (last \\boxed{} content), or None."""
    return last_boxed_content(completion)


def has_valid_format(completion):
    """True if the completion ends with a boxed answer that math-verify can parse."""
    boxed = last_boxed_content(completion)
    if not boxed:
        return False
    try:
        return len(parse(f"${boxed}$")) > 0
    except Exception:
        return False


def same_answer(a, b):
    """True if two extracted answers are mathematically equivalent."""
    if a is None or b is None:
        return False
    if a.strip() == b.strip():
        return True
    try:
        return bool(verify(parse(f"${a}$"), parse(f"${b}$")))
    except Exception:
        return False
