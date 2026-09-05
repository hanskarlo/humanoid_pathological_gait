#!/usr/bin/env bash
# Train N seeds of PPO+AMP and capture each one's evaluation rollout -- the multi-seed
# training-curve figure and per-seed metrics tables for the paper come from this.
#
# Usage:
#   ./scripts/train_and_evaluate_seeds.sh                          # 3 seeds, defaults below
#   SEEDS="1 2 3 4 5" MAX_ITERATIONS=1500 ./scripts/train_and_evaluate_seeds.sh
#
# Env vars (all optional):
#   SEEDS           space-separated seed list                (default: "1 2 3")
#   NUM_ENVS        parallel environments per run             (default: 14720)
#   MAX_ITERATIONS  PPO iterations per seed                   (default: 1200)
#   LOG_ROOT        where seed_<n>/ directories are written   (default: logs/paper)
#   EVAL_ENVS       environments in the post-training rollout (default: 64)
#   EVAL_STEPS      control steps in that rollout             (default: 1000, ~11 gait
#                   cycles at the measured 1.83 s stride -- see evaluate.py's undersampling
#                   warning if this is lowered)
#
# On gait-phase aliasing: the control step is 20 ms and the reference stride is now the
# stride's own measured duration, so stride_duration_s / dt is 91.5 rather than the exact
# 60 that silently collapsed cycle binning to 60 distinct phases in the 2026-09-04
# baseline. Nothing to do; noted because a future reference with a round duration would
# reintroduce it.
#
# Sequential, not parallel: one 14720-environment run already saturates the GPU, so
# running two at once only makes both slower and risks an out-of-memory kill hours in.
#
# Resumable. Re-running this script checks each seed's run directory for an existing
# checkpoint and, if one is there, passes --resume so train_amp.py continues from its
# newest model_*.pt instead of restarting at iteration 0. A seed already at
# MAX_ITERATIONS resumes into a no-op loop and proceeds straight to evaluation.
#
# Output, per seed, under $LOG_ROOT/seed_<n>/:
#   model_<iter>.pt, metrics.csv, run_config.json, a TensorBoard event file, and an
#   evaluation/ directory (rollout.npz, gait_cycle.npz, metrics.json, metrics.csv) written
#   by evaluate.py against the final checkpoint. Console output for each phase is
#   duplicated to $LOG_ROOT/seed_<n>.train.log and .eval.log.
#
# Draw the figures afterwards from the repository root (no Isaac Sim needed):
#   python3 -m scripts.plotting.plot_training_curves --run <LOG_ROOT>/seed_1:PPO-AMP ...
#   python3 -m scripts.plotting.plot_gait_cycle --data <LOG_ROOT>/seed_1/evaluation/gait_cycle.npz
#   python3 -m scripts.plotting.plot_stability  --eval_dir <LOG_ROOT>/seed_1/evaluation
set -uo pipefail

SEEDS=${SEEDS:-"1 2 3"}
NUM_ENVS=${NUM_ENVS:-14720}
MAX_ITERATIONS=${MAX_ITERATIONS:-1200}
LOG_ROOT=${LOG_ROOT:-logs/paper}
EVAL_STEPS=${EVAL_STEPS:-1000}
EVAL_ENVS=${EVAL_ENVS:-64}

export OMNI_KIT_ACCEPT_EULA=YES
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$LOG_ROOT"

echo "seeds=[$SEEDS] envs=$NUM_ENVS iterations=$MAX_ITERATIONS -> $LOG_ROOT"

for seed in $SEEDS; do
  run_dir="$LOG_ROOT/seed_${seed}"
  echo "=== seed $seed -> $run_dir ==="

  # --resume only on a restart: train_amp.py's --resume exits with an error if the
  # directory holds no checkpoint yet, which is exactly the state of a fresh seed.
  resume_args=()
  if compgen -G "$run_dir/model_*.pt" > /dev/null 2>&1; then
      echo "  found existing checkpoint(s) in $run_dir; resuming"
      resume_args=(--resume "$run_dir")
  fi

  uv run python scripts/train_amp.py \
      --num_envs "$NUM_ENVS" \
      --max_iterations "$MAX_ITERATIONS" \
      --seed "$seed" \
      --run_dir "$run_dir" \
      "${resume_args[@]}" \
      --headless 2>&1 | tee "$run_dir.train.log"

  checkpoint="$run_dir/model_${MAX_ITERATIONS}.pt"
  if [[ ! -f "$checkpoint" ]]; then
      echo "!! seed $seed produced no final checkpoint (crashed or was interrupted); skipping its evaluation"
      echo "   re-running this script will resume seed $seed from its newest checkpoint"
      continue
  fi

  # Capture straight away, while the conditions that produced the policy are known.
  uv run python scripts/evaluate.py \
      --checkpoint "$checkpoint" \
      --num_envs "$EVAL_ENVS" \
      --num_steps "$EVAL_STEPS" \
      --label "seed_${seed}" \
      --headless 2>&1 | tee "$run_dir.eval.log"
done

echo
echo "=== done. Draw the figures from the repository root: ==="
printf '  python3 -m scripts.plotting.plot_training_curves'
for seed in $SEEDS; do printf ' \\\n      --run humanoid_pathological_gait/%s/seed_%s:PPO-AMP' "$LOG_ROOT" "$seed"; done
printf ' \\\n      --double_column\n'
