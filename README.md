# Failure Routing (COMS 4705 Final Project)

Can gradient routing localize what a language model learns from failure?
We train Qwen2.5-Math-1.5B with GRPO on MATH, routing updates from incorrect
rollouts into a failure adapter and updates from correct rollouts into a
success adapter. Ceyda Topcu and Jake Li.

## What's here

| Path | What it does |
|---|---|
| `src/data.py` | Loads MATH train (with a fixed 500-problem dev split) and MATH500 |
| `src/checker.py` | Answer checking with Hugging Face `math-verify` (the training reward) |
| `src/metrics.py` | pass@k (unbiased estimator), confident-error rate, format validity |
| `scripts/eval_baseline.py` | Samples 16 solutions per problem and scores them |
| `src/adapters.py` | Success (S) and failure (F) adapters attached to every block of a frozen model |
| `src/grpo.py` | GRPO loss, group advantages, and gradient routing (modes: grpo, outcome, random, psr, nsr) |
| `scripts/train_grpo.py` | Trains the adapters with GRPO and routing; logs to `runs/<name>/` |
| `scripts/eval_adapters.py` | Evaluates a trained run with each adapter switched on/off (the ablation) |
| `scripts/plot_training.py` | Plots reward, format validity, and generation length over training |
| `tests/` | Tests for the checker, metrics, adapters, GRPO loss and routing |
| `kaggle_setup.ipynb` | Notebook that runs everything on Kaggle |

Code we did not write: `math-verify` (answer checking), `vllm` (fast sampling),
`transformers` and `datasets` (Hugging Face). Everything in `src/`, `scripts/`
and `tests/` is ours.

## Setup, step by step

### 1. Get the code

The code lives at https://github.com/ceydat709/coms4705-failure-routing.
Ask Ceyda to add you under **Settings → Collaborators** so you can push.

### 2. (Only if the repo is made private) Make a GitHub token for Kaggle

1. GitHub → your profile picture → **Settings → Developer settings →
   Personal access tokens → Fine-grained tokens → Generate new token**.
2. Under **Repository access**, choose **Only select repositories** and pick
   `coms4705-failure-routing`. Under **Permissions → Contents**, choose **Read-only**.
3. Generate it, copy it, and add it in Kaggle as a secret named `GITHUB_TOKEN`.

### 3. Set up Kaggle (each person)

1. Create an account at kaggle.com.
2. **Verify your phone number** (profile picture → Settings → Phone
   verification). Kaggle won't give you GPUs or internet access without it.

### 4. Create the notebook

1. Kaggle → **Create → New Notebook**, then **File → Import Notebook** and
   upload `kaggle_setup.ipynb`.
2. In the right sidebar under **Session options**, set
   **Accelerator = GPU T4 x2** and **Internet = On**.
3. Only if the repo is private: **Add-ons → Secrets → Add a new secret**,
   name it `GITHUB_TOKEN`, paste the token, and switch it on for this notebook.

### 5. Run it

Run the cells in order:

1. Check the GPU.
2. Clone the repo.
3. Install packages. This takes a few minutes.
4. Run the tests. All 7 should pass.
5. Check the data. It should print about `train=7000 dev=500 math500=500`.
6. Run the smoke test: 20 problems, 4 samples each.
7. Run the full baselines.

For the full baselines (step 7), use **Save Version → Save & Run All (Commit)**
so the run continues after you close the tab. Each full run is 500 problems ×
16 samples. Use the smoke test's timing to estimate how long it will take.

## Notes

- **Precision:** Kaggle's T4 GPUs don't support bfloat16, so the script
  automatically uses float16 there and bfloat16 on A100/L4 (GCP).
- **If vLLM won't install or crashes:** add `--backend hf` to use plain
  Hugging Face generation. It's much slower, so use `--limit` and fewer
  `--n_samples` while debugging.
- **Kaggle limits:** there's a weekly GPU quota, so check it under your
  profile. Stop the session when you're done to avoid wasting hours.
- **Moving to GCP later:** `git clone` the same repo on the VM, then run the
  same commands without the Kaggle-specific cells.

## Training (GRPO with routing)

```bash
# quick check that training runs (a few minutes)
python scripts/train_grpo.py --mode outcome --steps 3 --prompts_per_step 2 --group_size 4 \
    --max_new_tokens 256 --micro_batch 2 --name debug

# real runs (L4/A100 on GCP)
python scripts/train_grpo.py --mode grpo    --steps 50 --name grpo_seed0
python scripts/train_grpo.py --mode outcome --steps 50 --name outcome_seed0

# ablation: switch each adapter off
python scripts/eval_adapters.py --run runs/outcome_seed0 --split dev --limit 100 --n_samples 8

# training curves
python scripts/plot_training.py runs/grpo_seed0 runs/outcome_seed0 --out figures/training.png
```

Routing modes: `grpo` (both adapters get every update), `outcome` (wrong answers update
only F, right answers only S), `random` (labels shuffled within the batch, the control),
`psr`, `nsr`. On a T4 the script trains in float32; on A100/L4 it uses bfloat16.
