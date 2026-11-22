"""Evaluation metrics.

For each problem we draw n samples (16 in our setup) and record, per sample,
whether it is correct and what final answer it gave.

- pass@k: unbiased estimator from Chen et al. (2021), matching Zhu et al. (2025).
- confident-error rate: fraction of problems where at least `threshold` of the
  n samples agree on the same WRONG answer.
- format validity: fraction of samples with a parseable boxed answer.
"""

from math import comb


def pass_at_k(n, c, k):
    """Unbiased pass@k for one problem with n samples, c of them correct.

    pass@k = 1 - C(n - c, k) / C(n, k)
    """
    if k > n:
        raise ValueError(f"k={k} cannot exceed the number of samples n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def mean_pass_at_k(per_problem_correct, k):
    """Average pass@k over problems. per_problem_correct is a list of lists of bools."""
    scores = [pass_at_k(len(c), sum(c), k) for c in per_problem_correct]
    return sum(scores) / len(scores)


def largest_wrong_cluster(answers, correct, same_answer):
    """Size of the largest group of samples that share the same wrong answer.

    answers: list of extracted answers (str or None), one per sample
    correct: list of bools, one per sample
    same_answer: function deciding whether two answers are equivalent
    """
    wrong = [a for a, c in zip(answers, correct) if not c and a is not None]
    clusters = []  # list of [representative, count]
    for a in wrong:
        for cluster in clusters:
            if same_answer(cluster[0], a):
                cluster[1] += 1
                break
        else:
            clusters.append([a, 1])
    return max((count for _, count in clusters), default=0)


def confident_error_rate(per_problem_wrong_cluster, threshold):
    """Fraction of problems whose largest same-wrong-answer group has >= threshold samples."""
    hits = [size >= threshold for size in per_problem_wrong_cluster]
    return sum(hits) / len(hits)


def format_validity(per_problem_valid):
    """Fraction of all samples with a parseable boxed answer."""
    flat = [v for problem in per_problem_valid for v in problem]
    return sum(flat) / len(flat)
