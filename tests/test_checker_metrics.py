"""Sanity tests for the answer checker and metrics. Run: python -m pytest tests/"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from checker import extract_answer, has_valid_format, is_correct, same_answer  # noqa: E402
from data import last_boxed_content  # noqa: E402
from metrics import (  # noqa: E402
    confident_error_rate,
    largest_wrong_cluster,
    mean_pass_at_k,
    pass_at_k,
)


def test_last_boxed_handles_nesting():
    assert last_boxed_content("so \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"
    assert last_boxed_content("\\boxed{1} then \\boxed{2}") == "2"
    assert last_boxed_content("no answer here") is None
    assert last_boxed_content("truncated \\boxed{\\frac{1}{2") is None


def test_checker_equivalent_forms():
    assert is_correct("The answer is \\boxed{0.5}.", "\\frac{1}{2}")
    assert is_correct("Thus \\boxed{\\frac{1}{2}}", "\\frac12")
    assert is_correct("So n = 5. \\boxed{5}", "5")
    assert is_correct("\\boxed{2\\sqrt{2}}", "\\sqrt{8}")


def test_checker_rejects_wrong_or_missing():
    assert not is_correct("The answer is \\boxed{3}.", "5")
    assert not is_correct("I am not sure.", "5")
    assert not is_correct("", "5")


def test_format_validity():
    assert has_valid_format("so \\boxed{42}")
    assert not has_valid_format("no box")
    assert not has_valid_format("\\boxed{")


def test_same_answer():
    assert same_answer("0.5", "\\frac{1}{2}")
    assert not same_answer("3", "4")
    assert extract_answer("x \\boxed{7}") == "7"


def test_pass_at_k_known_values():
    assert pass_at_k(16, 0, 1) == 0.0
    assert pass_at_k(16, 16, 1) == 1.0
    assert abs(pass_at_k(16, 4, 1) - 0.25) < 1e-9
    assert pass_at_k(16, 1, 16) == 1.0
    # one correct out of 16, k=2: 1 - C(15,2)/C(16,2) = 1 - 105/120
    assert abs(pass_at_k(16, 1, 2) - (1 - 105 / 120)) < 1e-9
    assert abs(mean_pass_at_k([[True, False], [False, False]], 1) - 0.25) < 1e-9


def test_confident_error():
    answers = ["3"] * 12 + ["5"] * 4
    correct = [False] * 12 + [True] * 4
    size = largest_wrong_cluster(answers, correct, same_answer)
    assert size == 12
    assert confident_error_rate([12, 3], threshold=12) == 0.5
    # equivalent wrong answers are grouped together
    assert largest_wrong_cluster(["0.5", "\\frac{1}{2}", "2"], [False] * 3, same_answer) == 2
