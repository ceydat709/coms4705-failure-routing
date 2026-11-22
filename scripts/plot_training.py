"""Plot training curves (reward, format validity, generation length) for one or more runs.

Example:
    python scripts/plot_training.py runs/grpo_seed0 runs/outcome_seed0 --out figures/training.png
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def smooth(xs, window):
    if window <= 1:
        return xs
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        out.append(sum(xs[lo: i + 1]) / (i + 1 - lo))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="figures/training.png")
    p.add_argument("--smooth", type=int, default=5)
    args = p.parse_args()

    panels = [("reward", "Training reward (accuracy)"),
              ("format_validity", "Format validity"),
              ("mean_gen_tokens", "Mean generated tokens")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.2))
    for run in args.runs:
        rows = [json.loads(line) for line in open(os.path.join(run, "log.jsonl"))]
        steps = [r["step"] for r in rows]
        for ax, (key, _) in zip(axes, panels):
            ax.plot(steps, smooth([r[key] for r in rows], args.smooth), label=os.path.basename(run.rstrip("/")))
    for ax, (_, title) in zip(axes, panels):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
