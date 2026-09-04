#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Helper script to find the optimal maximum --envs that fits in VRAM.
# It incrementally increases the environment count and runs a very short
# 3-iteration test. If it crashes with a CUDA OOM (or hangs, or is killed
# by the kernel for using too much memory), it steps back, then bisects
# between the last size that fit and the first that didn't.
#
# Scene setup time grows with --envs, so a single test can take anywhere from
# ~15s (small counts) to several minutes (tens of thousands of envs). Each
# test is bounded by TEST_TIMEOUT_S below so a driver-level hang can't stall
# the script forever; override it with an env var if your GPU needs longer.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p logs

TEST_TIMEOUT_S="${TEST_TIMEOUT_S:-600}"

echo "================================================="
echo " Starting VRAM Capacity Test for --envs"
echo "================================================="
echo "This will iteratively launch the environment to find the limit."
echo "Each test is capped at ${TEST_TIMEOUT_S}s; larger --envs take longer to set up."
echo ""

# The sizes we want to test
ENVS_TO_TEST=(512 1024 2048 4096 6144 8192 12288 16384)
MAX_SAFE=""
FIRST_FAIL=""

# Runs one short training smoke test at $1 envs and reports the verdict.
# Prints one of: pass | oom | timeout | crash
# The exit code additionally carries the same information (0/1/2/3) so
# callers can branch without re-parsing the printed word.
test_envs() {
    local envs="$1"
    local log_file="logs/vram_test_${envs}.log"

    set +e
    timeout "${TEST_TIMEOUT_S}" scripts/train.sh --envs "${envs}" --iters 3 > "${log_file}" 2>&1
    local exit_code=$?
    set -e

    if [ "${exit_code}" -eq 124 ]; then
        echo "timeout"
        return 3
    fi

    # Besides PyTorch's own message, PhysX/Isaac Sim and the kernel OOM killer
    # can all end a run without ever printing "CUDA out of memory": the kernel
    # sends SIGKILL (exit 137) with nothing in stdout/stderr, and PhysX/driver
    # failures use their own wording. Treat all of these as OOM, not just the
    # literal CUDA string.
    if grep -qiE "cuda out of memory|allocation failed|out of memory|cudaErrorMemoryAllocation|failed to allocate" "${log_file}" \
        || [ "${exit_code}" -eq 137 ]; then
        echo "oom"
        return 1
    elif [ "${exit_code}" -ne 0 ]; then
        echo "crash"
        return 2
    fi

    echo "pass"
    return 0
}

for ENVS in "${ENVS_TO_TEST[@]}"; do
    echo "▶️ Testing with --envs ${ENVS}..."

    set +e
    RESULT="$(test_envs "${ENVS}")"
    STATUS=$?
    set -e

    case "${STATUS}" in
        0)
            echo "✅ PASSED: --envs ${ENVS} fits in VRAM!"
            MAX_SAFE="${ENVS}"
            ;;
        1)
            echo "❌ FAILED: --envs ${ENVS} exceeded VRAM (CUDA OOM)."
            FIRST_FAIL="${ENVS}"
            break
            ;;
        3)
            echo "⏱️ FAILED: --envs ${ENVS} did not finish within ${TEST_TIMEOUT_S}s (likely hung near the VRAM limit)."
            echo "Check logs/vram_test_${ENVS}.log for details, or raise TEST_TIMEOUT_S if this size is just slow to set up."
            FIRST_FAIL="${ENVS}"
            break
            ;;
        *)
            echo "⚠️ FAILED: --envs ${ENVS} crashed for an unknown reason."
            echo "Check logs/vram_test_${ENVS}.log for details."
            FIRST_FAIL="${ENVS}"
            break
            ;;
    esac
    echo "-------------------------------------------------"
done

# Coarse grid found a pass/fail boundary: bisect between them to narrow it down
# instead of reporting the coarse MAX_SAFE, which can undersell the GPU by
# thousands of environments (e.g. a real gap between 8192 and 12288).
if [ -n "${MAX_SAFE}" ] && [ -n "${FIRST_FAIL}" ]; then
    echo ""
    echo "Narrowing down the boundary between ${MAX_SAFE} (fits) and ${FIRST_FAIL} (fails)..."
    LOW="${MAX_SAFE}"
    HIGH="${FIRST_FAIL}"
    # Stop once the bracket is tight enough that further precision isn't useful.
    while [ $((HIGH - LOW)) -gt 128 ]; do
        MID=$(( (LOW + HIGH) / 2 ))
        MID=$(( (MID / 64) * 64 ))
        if [ "${MID}" -le "${LOW}" ]; then
            break
        fi
        echo "▶️ Testing with --envs ${MID}..."
        set +e
        RESULT="$(test_envs "${MID}")"
        STATUS=$?
        set -e
        if [ "${STATUS}" -eq 0 ]; then
            echo "✅ PASSED: --envs ${MID} fits in VRAM!"
            LOW="${MID}"
            MAX_SAFE="${MID}"
        else
            echo "❌ FAILED: --envs ${MID} (${RESULT})."
            HIGH="${MID}"
        fi
        echo "-------------------------------------------------"
    done
fi

echo ""
echo "================================================="
if [ -z "${MAX_SAFE}" ]; then
    echo "All tests failed. Check if your GPU supports Isaac Sim."
else
    echo "🎉 MAXIMUM SAFE ENVIRONMENTS: ${MAX_SAFE}"

    # Recommend headroom below the measured edge rather than the edge itself:
    # a full multi-thousand-iteration run can transiently use more memory than
    # this 3-iteration smoke test does (e.g. checkpoint/eval spikes).
    RECOMMENDED=$(( (MAX_SAFE * 9 / 10 / 64) * 64 ))
    [ "${RECOMMENDED}" -lt 64 ] && RECOMMENDED="${MAX_SAFE}"

    echo "You can use this value for your training runs:"
    echo "  scripts/train.sh --envs ${MAX_SAFE} --iters 3500"
    echo "For a long production run, ${RECOMMENDED} (~90% of the measured max) leaves headroom"
    echo "for the memory a 3-iteration smoke test won't show."

    if [ "${MAX_SAFE}" == "${ENVS_TO_TEST[-1]}" ]; then
        echo "Note: You maxed out the test! You might be able to go even higher."
    fi
fi
echo "================================================="
