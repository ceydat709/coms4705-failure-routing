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
| `tests/` | Sanity tests for the checker and metrics |
| `kaggle_setup.ipynb` | Notebook that runs everything on Kaggle |

Code we did not write: `math-verify` (answer checking), `vllm` (fast sampling),
`transformers` and `datasets` (Hugging Face). Everything in `src/`, `scripts/`
and `tests/` is ours.

## Setup, step by step

### 1. Put the code on GitHub (one person, once)

1. Go to github.com → **New repository**. Name it `failure-routing`, set it to
   **Private**, and create it without a README.
2. Open the repository page and click **uploading an existing file**. Drag in
   everything from this folder, including the `src`, `scripts` and `tests`
   folders, then click **Commit changes**.
3. Go to **Settings → Collaborators → Add people** and add your teammate.

### 2. Make a GitHub token so Kaggle can read the private repo

1. GitHub → your profile picture → **Settings → Developer settings →
   Personal access tokens → Fine-grained tokens → Generate new token**.
2. Under **Repository access**, choose **Only select repositories** and pick
   `failure-routing`. Under **Permissions → Contents**, choose **Read-only**.
3. Generate it and copy the token. You won't see it again.

### 3. Set up Kaggle (each person)

1. Create an account at kaggle.com.
2. **Verify your phone number** (profile picture → Settings → Phone
   verification). Kaggle won't give you GPUs or internet access without it.

### 4. Create the notebook

1. Kaggle → **Create → New Notebook**, then **File → Import Notebook** and
   upload `kaggle_setup.ipynb`.
2. In the right sidebar under **Session options**, set
   **Accelerator = GPU T4 x2** and **Internet = On**.
3. **Add-ons → Secrets → Add a new secret**: name it `GITHUB_TOKEN`, paste the
   token, and switch it on for this notebook.
4. In the second code cell, replace `YOUR_GITHUB_USERNAME` with the username
   that owns the repo.

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
