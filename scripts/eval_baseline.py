"""Evaluate a model on MATH500 (or our dev set) with n samples per problem.

Produces the milestone's prompting baselines (zero-shot and 4-shot), and is
reused later to evaluate trained adapter configurations.

Examples (run from the repo root):
    # quick smoke test on 20 problems
    python scripts/eval_baseline.py --shots 0 --limit 20 --name smoke

    # full zero-shot and 4-shot baselines on MATH500
    python scripts/eval_baseline.py --shots 0 --name base_0shot
    python scripts/eval_baseline.py --shots 4 --name base_4shot

Outputs go to results/<name>/: samples.jsonl (every sample) and summary.json.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from checker import extract_answer, has_valid_format, is_correct, same_answer  # noqa: E402
from data import load_math500, load_math_train_dev  # noqa: E402
from metrics import (  # noqa: E402
    confident_error_rate,
    format_validity,
    largest_wrong_cluster,
    mean_pass_at_k,
)

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def pick_fewshot_examples(train, n_shots, seed=0, max_solution_chars=700):
    """Fixed few-shot examples from the training split, with short solutions
    so the prompt fits in the model's context window."""
    pool = [ex for ex in train if len(ex["solution"]) <= max_solution_chars]
    rng = random.Random(seed)
    return rng.sample(pool, n_shots)


def build_prompt(tokenizer, problem, shots):
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in shots:
            messages.append({"role": "user", "content": ex["problem"]})
            messages.append({"role": "assistant", "content": ex["solution"]})
        messages.append({"role": "user", "content": problem})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Plain-text fallback for tokenizers without a chat template.
    parts = [SYSTEM_PROMPT, ""]
    for ex in shots:
        parts += [f"Problem: {ex['problem']}", f"Solution: {ex['solution']}", ""]
    parts += [f"Problem: {problem}", "Solution:"]
    return "\n".join(parts)


def pick_dtype():
    """bfloat16 needs compute capability 8.0+ (A100, L4). T4s (Kaggle, Colab) are 7.5.

    Note: torch.cuda.is_bf16_supported() can return True on a T4 (emulation),
    but vLLM rejects bf16 there, so we check the compute capability directly.
    """
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return "bfloat16"
    return "float16"


def generate_vllm(args, prompts, dtype):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype=dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
    )
    params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, params)
    return [[o.text for o in out.outputs] for out in outputs]


def generate_hf(args, prompts, dtype):
    """Slower fallback using plain Hugging Face transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, dtype), device_map="auto"
    )
    model.eval()
    all_texts = []
    for i in range(0, len(prompts), args.hf_batch_size):
        batch = prompts[i : i + args.hf_batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=args.temperature,
                top_p=1.0,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.n_samples,
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
        for j in range(len(batch)):
            all_texts.append(texts[j * args.n_samples : (j + 1) * args.n_samples])
        print(f"  generated {min(i + args.hf_batch_size, len(prompts))}/{len(prompts)} problems")
    return all_texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--split", choices=["math500", "dev"], default="math500")
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N problems")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--hf_batch_size", type=int, default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="auto",
                   help="auto picks bfloat16 on A100/L4 and float16 on T4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", required=True, help="output folder name under results/")
    args = p.parse_args()

    from transformers import AutoTokenizer

    random.seed(args.seed)
    train, dev = load_math_train_dev()
    problems = load_math500() if args.split == "math500" else dev
    if args.limit:
        problems = problems[: args.limit]
    shots = pick_fewshot_examples(train, args.shots) if args.shots else []

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [build_prompt(tokenizer, ex["problem"], shots) for ex in problems]
    dtype = pick_dtype() if args.dtype == "auto" else args.dtype
    print(f"{len(problems)} problems, {args.n_samples} samples each, dtype={dtype}, backend={args.backend}")

    start = time.time()
    gen = generate_vllm if args.backend == "vllm" else generate_hf
    completions = gen(args, prompts, dtype)
    gen_minutes = (time.time() - start) / 60
    print(f"Generation took {gen_minutes:.1f} min. Scoring...")

    out_dir = os.path.join("results", args.name)
    os.makedirs(out_dir, exist_ok=True)
    per_correct, per_valid, per_cluster = [], [], []
    by_level, by_subject = defaultdict(list), defaultdict(list)

    with open(os.path.join(out_dir, "samples.jsonl"), "w") as f:
        for ex, samples in zip(problems, completions):
            correct = [is_correct(s, ex["answer"]) for s in samples]
            answers = [extract_answer(s) for s in samples]
            valid = [has_valid_format(s) for s in samples]
            cluster = largest_wrong_cluster(answers, correct, same_answer)
            per_correct.append(correct)
            per_valid.append(valid)
            per_cluster.append(cluster)
            by_level[str(ex["level"])].append(correct)
            by_subject[ex["subject"]].append(correct)
            f.write(json.dumps({**ex, "completions": samples, "correct": correct,
                                "answers": answers, "valid": valid,
                                "largest_wrong_cluster": cluster}) + "\n")

    ks = [k for k in (1, 2, 4, 8, 16) if k <= args.n_samples]
    summary = {
        "config": vars(args),
        "n_problems": len(problems),
        "generation_minutes": round(gen_minutes, 1),
        "pass@k": {k: round(mean_pass_at_k(per_correct, k), 4) for k in ks},
        "confident_error_rate": {
            t: round(confident_error_rate(per_cluster, t), 4)
            for t in range(max(2, args.n_samples // 2), args.n_samples + 1)
        },
        "format_validity": round(format_validity(per_valid), 4),
        "pass@1_by_level": {lv: round(mean_pass_at_k(c, 1), 4) for lv, c in sorted(by_level.items())},
        "pass@1_by_subject": {s: round(mean_pass_at_k(c, 1), 4) for s, c in sorted(by_subject.items())},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("pass@k", "format_validity")}, indent=2))
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
