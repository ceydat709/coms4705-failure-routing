"""Evaluate a trained run with each adapter switched on or off (the ablation).

For each configuration (both adapters, success only, failure only, base model)
we sample n solutions per problem and report pass@k, confident-error rate and
format validity, using the same metrics as eval_baseline.py.

Example:
    python scripts/eval_adapters.py --run runs/outcome_seed0 --split dev --limit 100 --n_samples 8

Writes runs/<name>/eval_<split>_step<k>.json.
"""

import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from adapters import attach_adapters, load_adapter_state_dict, set_adapter_scales  # noqa: E402
from checker import extract_answer, has_valid_format, is_correct, same_answer  # noqa: E402
from data import load_math500, load_math_train_dev  # noqa: E402
from metrics import confident_error_rate, format_validity, largest_wrong_cluster, mean_pass_at_k  # noqa: E402
from train_grpo import build_prompt, pick_dtype, rollout  # noqa: E402

CONFIGS = {
    "S+F": (1.0, 1.0),
    "S_only": (1.0, 0.0),   # failure adapter removed
    "F_only": (0.0, 1.0),   # success adapter removed
    "base": (0.0, 0.0),
}


def latest_checkpoint(run_dir):
    ckpts = glob.glob(os.path.join(run_dir, "adapters_step*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no adapters_step*.pt in {run_dir}")
    return max(ckpts, key=lambda p: int(re.search(r"step(\d+)", p).group(1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="runs/<name> directory from train_grpo.py")
    p.add_argument("--checkpoint", default=None, help="default: latest adapters_step*.pt")
    p.add_argument("--split", choices=["dev", "math500"], default="dev")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--batch_problems", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=None, help="default: the run's setting")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--configs", default="S+F,S_only,F_only,base")
    p.add_argument("--scale_F", type=float, default=None,
                   help="optional: also evaluate S plus F scaled by this factor")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    ckpt = args.checkpoint or latest_checkpoint(args.run)
    step = int(re.search(r"step(\d+)", ckpt).group(1))
    max_new = args.max_new_tokens or cfg["max_new_tokens"]
    torch.manual_seed(args.seed)

    _, dev = load_math_train_dev()
    problems = (load_math500() if args.split == "math500" else dev)[: args.limit]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = pick_dtype(args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["model"], padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype).to(device)
    attach_adapters(model, width=cfg["adapter_width"])
    load_adapter_state_dict(model, torch.load(ckpt, map_location="cpu"))

    configs = {name: CONFIGS[name] for name in args.configs.split(",")}
    if args.scale_F is not None:
        configs[f"S+{args.scale_F}F"] = (1.0, args.scale_F)

    results = {"run": args.run, "checkpoint": ckpt, "step": step, "split": args.split,
               "n_problems": len(problems), "n_samples": args.n_samples, "configs": {}}
    for name, (sS, sF) in configs.items():
        t0 = time.time()
        set_adapter_scales(model, sS, sF)
        per_correct, per_valid, per_cluster = [], [], []
        for i in range(0, len(problems), args.batch_problems):
            chunk = problems[i: i + args.batch_problems]
            prompts = [build_prompt(tok, ex["problem"]) for ex in chunk]
            _, _, _, _, texts, _ = rollout(model, tok, prompts, args.n_samples, max_new, args.temperature)
            for j, ex in enumerate(chunk):
                samples = texts[j * args.n_samples: (j + 1) * args.n_samples]
                correct = [is_correct(s, ex["answer"]) for s in samples]
                answers = [extract_answer(s) for s in samples]
                per_correct.append(correct)
                per_valid.append([has_valid_format(s) for s in samples])
                per_cluster.append(largest_wrong_cluster(answers, correct, same_answer))
        n = args.n_samples
        ks = [k for k in (1, 2, 4, 8, 16) if k <= n]
        results["configs"][name] = {
            "scales": [sS, sF],
            "pass@k": {k: round(mean_pass_at_k(per_correct, k), 4) for k in ks},
            "confident_error_rate": {t: round(confident_error_rate(per_cluster, t), 4)
                                     for t in range(max(2, n // 2), n + 1)},
            "format_validity": round(format_validity(per_valid), 4),
            "minutes": round((time.time() - t0) / 60, 1),
        }
        r = results["configs"][name]
        print(f"{name:8s} pass@1 {r['pass@k'][1]:.3f}  pass@{n} {r['pass@k'][n]:.3f}  "
              f"valid {r['format_validity']:.3f}  ({r['minutes']} min)", flush=True)

    out = os.path.join(args.run, f"eval_{args.split}_step{step}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
