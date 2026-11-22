"""Data loading for MATH (training) and MATH500 (final evaluation).

- Training data: the MATH training split (7,500 problems, 7 subjects), from
  the Hugging Face dataset EleutherAI/hendrycks_math (one config per subject).
- Development set: 500 problems held out from the training split with a fixed
  seed, used for checkpoint selection and tuning.
- Test data: MATH500 (HuggingFaceH4/MATH-500), a 500-problem subset of the MATH
  test split. Evaluate on it only at the end.

Each example is returned as a dict with keys:
    problem, answer, subject, level, solution
"""

import random

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

DEV_SIZE = 500
SPLIT_SEED = 4705


def last_boxed_content(text):
    """Return the contents of the last \\boxed{...} (or \\fbox{...}) in text.

    Handles nested braces, e.g. \\boxed{\\frac{1}{2}} -> \\frac{1}{2}.
    Returns None if there is no complete boxed expression.
    """
    if text is None:
        return None
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None  # unbalanced braces, e.g. a truncated generation


def _normalize_train_example(ex):
    for key in ("problem", "solution", "level", "type"):
        if key not in ex:
            raise KeyError(
                f"Expected field '{key}' in MATH example, found {list(ex.keys())}. "
                "The dataset schema may have changed; update _normalize_train_example."
            )
    return {
        "problem": ex["problem"],
        "solution": ex["solution"],
        "answer": last_boxed_content(ex["solution"]),
        "subject": ex["type"],
        "level": ex["level"],
    }


def load_math_train_dev(dev_size=DEV_SIZE, seed=SPLIT_SEED):
    """Load the MATH training split and hold out a fixed development set.

    Returns (train, dev): two lists of example dicts.
    """
    from datasets import load_dataset

    examples = []
    for subject in MATH_SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        examples.extend(_normalize_train_example(ex) for ex in ds)

    # Drop the few problems whose reference solution has no \boxed answer.
    n_before = len(examples)
    examples = [ex for ex in examples if ex["answer"]]
    if len(examples) < n_before:
        print(f"Dropped {n_before - len(examples)} training problems with no boxed answer.")

    rng = random.Random(seed)
    rng.shuffle(examples)
    dev, train = examples[:dev_size], examples[dev_size:]
    return train, dev


def load_math500():
    """Load MATH500 for final evaluation."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for ex in ds:
        for key in ("problem", "answer", "subject", "level"):
            if key not in ex:
                raise KeyError(
                    f"Expected field '{key}' in MATH500 example, found {list(ex.keys())}."
                )
        out.append(
            {
                "problem": ex["problem"],
                "solution": ex.get("solution", ""),
                "answer": ex["answer"],
                "subject": ex["subject"],
                "level": ex["level"],
            }
        )
    return out


if __name__ == "__main__":
    train, dev = load_math_train_dev()
    test = load_math500()
    print(f"train={len(train)}  dev={len(dev)}  math500={len(test)}")
    print("Example:", train[0]["problem"][:200], "->", train[0]["answer"])
