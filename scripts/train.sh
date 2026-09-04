#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Train the pathological-gait policy with PPO + Adversarial Motion Prior.
#
# Usage: scripts/train.sh [--preset smoke|short|full] [--envs N] [--iters N]
#                         [--seed N] [--rsl-rl] [--gui] [-- <extra args>]
#
#   --preset smoke   64 envs,    20 iterations  -- proves the loop runs (~1 min)
#   --preset short  512 envs,   500 iterations  -- shape-of-the-curve check
#   --preset full  4096 envs,  3500 iterations  -- publication run (default)
#   --rsl-rl         Use stock RSL-RL PPO instead of PPO+AMP. Sanity check only:
#                    this rsl_rl has no AMP support, so the gait will not be shaped
#                    by the clinical motion prior.
#   --gui            Render the Isaac Sim viewport (much slower; for eyeballing only).
#
# Everything after `--` is forwarded verbatim to the underlying Python script.
# The run is tee'd to <log_dir>/train.log; scroll it or tail it live.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

PRESET=full
NUM_ENVS=""
MAX_ITERS=""
SEED=""
USE_RSL_RL=0
HEADLESS="--headless"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset) PRESET="${2:?--preset needs a value}"; shift 2 ;;
        --envs)   NUM_ENVS="${2:?--envs needs a value}"; shift 2 ;;
        --iters)  MAX_ITERS="${2:?--iters needs a value}"; shift 2 ;;
        --seed)   SEED="${2:?--seed needs a value}"; shift 2 ;;
        --rsl-rl) USE_RSL_RL=1; shift ;;
        --gui)    HEADLESS=""; shift ;;
        --)       shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '4,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) die "Unknown argument: $1 (use -- to forward args to the Python script)" ;;
    esac
done

case "${PRESET}" in
    smoke) DEF_ENVS=64;   DEF_ITERS=20 ;;
    short) DEF_ENVS=512;  DEF_ITERS=500 ;;
    full)  DEF_ENVS=4096; DEF_ITERS=3500 ;;
    *) die "Unknown preset '${PRESET}'. Choose smoke, short or full." ;;
esac
NUM_ENVS="${NUM_ENVS:-${DEF_ENVS}}"
MAX_ITERS="${MAX_ITERS:-${DEF_ITERS}}"

require_env
ensure_eula

if [[ "${USE_RSL_RL}" -eq 1 ]]; then
    ENTRY="${SCRIPT_DIR}/train_rsl_rl.py"
    LOG_ROOT="${EXT_ROOT}/logs/rsl_rl"
    warn "Stock RSL-RL has no AMP support: this trains reference tracking without the"
    warn "adversarial motion prior. Use it to sanity-check the environment, not to"
    warn "produce a gait policy."
else
    ENTRY="${SCRIPT_DIR}/train_amp.py"
    LOG_ROOT="${EXT_ROOT}/logs/ppo_amp"
fi

RUN_DIR="${LOG_ROOT}/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${RUN_DIR}"

# Pass the run directory explicitly so train.log sits beside this run's checkpoints
# instead of in a second directory timestamped a few seconds apart.
ARGS=(--num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITERS}" --run_dir "${RUN_DIR}")
[[ -n "${SEED}" ]] && ARGS+=(--seed "${SEED}")
[[ -n "${HEADLESS}" ]] && ARGS+=("${HEADLESS}")
# Guard the expansion: "${EXTRA[@]:-}" on an empty array appends a stray empty argument,
# which argparse then rejects.
[[ ${#EXTRA[@]} -gt 0 ]] && ARGS+=("${EXTRA[@]}")

info "preset ${PRESET}: ${NUM_ENVS} envs x ${MAX_ITERS} iterations"
info "algorithm: $([[ "${USE_RSL_RL}" -eq 1 ]] && echo 'RSL-RL PPO (sanity)' || echo 'PPO + AMP')"
info "log: ${RUN_DIR}/train.log"
printf '\n'

set +e
"${PY_RUN[@]}" "${ENTRY}" "${ARGS[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
STATUS="${PIPESTATUS[0]}"
set -e

printf '\n'
if [[ "${STATUS}" -ne 0 ]]; then
    die "Training exited with status ${STATUS}. Full output: ${RUN_DIR}/train.log"
fi

CKPT="$(latest_checkpoint "${RUN_DIR}")"
ok "Training finished."
[[ -n "${CKPT}" ]] && printf '  latest checkpoint: %s\n' "${CKPT}"
printf '  play it back:      scripts/play.sh\n'
