#!/usr/bin/env bash
# Milestone runs on one GPU (GCP L4): GRPO baseline vs. outcome routing,
# then adapter ablations and the training-curve figure.
# Usage (inside tmux):  bash scripts/run_milestone.sh 2>&1 | tee logs/milestone.log
set -u
mkdir -p logs figures

STEPS=${STEPS:-50}
TOK=${TOK:-512}
COMMON="--steps $STEPS --prompts_per_step 8 --group_size 8 --max_new_tokens $TOK --max_level 3 --save_every 25"

echo "== tests =="
python -m pytest -q tests || exit 1

echo "== 2-step debug run =="
python scripts/train_grpo.py --mode outcome --name debug --steps 2 --prompts_per_step 2 \
  --group_size 4 --max_new_tokens 128 --save_every 2 || exit 1

for MODE in grpo outcome; do
  echo "== train $MODE =="
  python scripts/train_grpo.py --mode $MODE --name $MODE $COMMON
done

for MODE in grpo outcome; do
  echo "== eval $MODE =="
  python scripts/eval_adapters.py --run runs/$MODE --split dev --limit 50 --n_samples 8
done

python scripts/plot_training.py runs/grpo runs/outcome --out figures/training.png
echo "== done =="
