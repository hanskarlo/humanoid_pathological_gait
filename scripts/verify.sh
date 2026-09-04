#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Preflight gate. Runs every check that can fail before a long training run does:
# task registration, a NaN-free zero-action rollout, the pathology assertions, and a
# 5-iteration stock-PPO run that exercises the full learning path.
#
# Usage: scripts/verify.sh [--quick]
#   --quick  Skip the RSL-RL training check (saves ~1 minute).

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

QUICK=0
for arg in "$@"; do
    case "${arg}" in
        --quick) QUICK=1 ;;
        -h|--help) sed -n '4,11p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) die "Unknown argument: ${arg}" ;;
    esac
done

require_env
ensure_eula

FAILED=()
run_check() {
    local name="$1"; shift
    info "${name}"
    if "$@"; then
        ok "${name}"
    else
        printf '%sfail%s %s\n' "${C_RED}" "${C_OFF}" "${name}" >&2
        FAILED+=("${name}")
    fi
}

run_check "task registration" \
    "${PY_RUN[@]}" "${SCRIPT_DIR}/list_envs.py" --show_presets

run_check "zero-action rollout is finite (training task)" \
    "${PY_RUN[@]}" "${SCRIPT_DIR}/zero_agent.py" --task Isaac-H1-Pathological-Gait-v0 \
        --num_envs 16 --num_steps 100 --headless

run_check "zero-action rollout is finite (play task)" \
    "${PY_RUN[@]}" "${SCRIPT_DIR}/zero_agent.py" --task Isaac-H1-Pathological-Gait-Play-v0 \
        --num_envs 16 --num_steps 100 --headless

run_check "pathology machinery is live" \
    "${PY_RUN[@]}" "${SCRIPT_DIR}/check_pathology.py" --num_envs 64 --headless

if [[ "${QUICK}" -eq 0 ]]; then
    run_check "environment drives a standard learner" \
        "${PY_RUN[@]}" "${SCRIPT_DIR}/train_rsl_rl.py" --num_envs 64 --max_iterations 5 \
            --seed 0 --log_dir "${EXT_ROOT}/logs/verify" --headless
fi

printf '\n'
if [[ ${#FAILED[@]} -eq 0 ]]; then
    printf '%sAll preflight checks passed.%s\n' "${C_GREEN}${C_BOLD}" "${C_OFF}"
    exit 0
fi
printf '%s%d check(s) failed:%s\n' "${C_RED}${C_BOLD}" "${#FAILED[@]}" "${C_OFF}"
printf '  - %s\n' "${FAILED[@]}"
exit 1
