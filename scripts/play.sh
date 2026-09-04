#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Play back a trained policy and print clinical gait metrics.
#
# Usage: scripts/play.sh [--checkpoint PATH] [--envs N] [--steps N]
#                        [--video] [--gui] [-- <extra args>]
#
#   --checkpoint  Defaults to the most recent checkpoint under logs/ppo_amp/.
#   --video       Record the rollout to <checkpoint dir>/videos/.
#   --gui         Render the Isaac Sim viewport instead of running headless.
#
# Playback uses the deterministic Isaac-H1-Pathological-Gait-Play-v0 task and the policy
# mean action, so repeated runs of one checkpoint are comparable.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

CKPT=""
NUM_ENVS=16
NUM_STEPS=600
VIDEO=""
HEADLESS="--headless"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CKPT="${2:?--checkpoint needs a value}"; shift 2 ;;
        --envs)       NUM_ENVS="${2:?--envs needs a value}"; shift 2 ;;
        --steps)      NUM_STEPS="${2:?--steps needs a value}"; shift 2 ;;
        --video)      VIDEO="--video"; shift ;;
        --gui)        HEADLESS=""; shift ;;
        --)           shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '4,16p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) die "Unknown argument: $1 (use -- to forward args to the Python script)" ;;
    esac
done

require_env
ensure_eula

if [[ -z "${CKPT}" ]]; then
    CKPT="$(latest_checkpoint "${EXT_ROOT}/logs/ppo_amp")"
    [[ -n "${CKPT}" ]] || die "No checkpoint found under logs/ppo_amp/. Train one first, or pass --checkpoint."
    info "using most recent checkpoint"
fi
[[ -f "${CKPT}" ]] || die "Checkpoint not found: ${CKPT}"

# Video capture needs a render pipeline, which the headless flag would switch off.
if [[ -n "${VIDEO}" && -n "${HEADLESS}" ]]; then
    info "recording video headless via the offscreen renderer"
fi

ARGS=(--checkpoint "${CKPT}" --num_envs "${NUM_ENVS}" --num_steps "${NUM_STEPS}")
[[ -n "${VIDEO}" ]] && ARGS+=("${VIDEO}")
[[ -n "${HEADLESS}" ]] && ARGS+=("${HEADLESS}")
[[ ${#EXTRA[@]} -gt 0 ]] && ARGS+=("${EXTRA[@]}")

info "checkpoint: ${CKPT}"
printf '\n'
exec "${PY_RUN[@]}" "${SCRIPT_DIR}/play.py" "${ARGS[@]}"
