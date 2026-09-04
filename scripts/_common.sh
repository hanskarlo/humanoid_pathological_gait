#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Shared setup sourced by every shell entry point in this directory.
# Not meant to be executed directly.

set -euo pipefail

# Resolve the extension root from this file's location, so the scripts work from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly SCRIPT_DIR EXT_ROOT

if [[ -t 1 ]]; then
    C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'; C_RED=$'\033[31m'
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_OFF=$'\033[0m'
else
    C_BOLD=''; C_DIM=''; C_RED=''; C_GREEN=''; C_YELLOW=''; C_OFF=''
fi
readonly C_BOLD C_DIM C_RED C_GREEN C_YELLOW C_OFF

info()  { printf '%s==>%s %s\n' "${C_BOLD}" "${C_OFF}" "$*"; }
ok()    { printf '%s  ok%s %s\n' "${C_GREEN}" "${C_OFF}" "$*"; }
warn()  { printf '%swarn%s %s\n' "${C_YELLOW}" "${C_OFF}" "$*" >&2; }
die()   { printf '%serr %s %s\n' "${C_RED}" "${C_OFF}" "$*" >&2; exit 1; }

# Isaac Sim blocks on an interactive EULA prompt, which hangs any headless or scripted run.
# Setting this variable is NVIDIA's documented way to accept it non-interactively; we only
# set it when the caller has not, and say so, rather than accepting a licence silently.
ensure_eula() {
    if [[ -z "${OMNI_KIT_ACCEPT_EULA:-}" ]]; then
        warn "OMNI_KIT_ACCEPT_EULA is unset; setting it to YES for this run."
        warn "That accepts the NVIDIA Omniverse licence agreement on your behalf:"
        warn "  https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html"
        warn "Export it yourself to silence this notice."
        export OMNI_KIT_ACCEPT_EULA=YES
    fi
}

# Prefer `uv run` so the project environment is resolved and kept in sync; fall back to the
# committed virtualenv when uv is unavailable (e.g. a CI image that only restores .venv).
resolve_python() {
    if command -v uv >/dev/null 2>&1; then
        PY_RUN=(uv run --project "${EXT_ROOT}" python)
    elif [[ -x "${EXT_ROOT}/.venv/bin/python" ]]; then
        PY_RUN=("${EXT_ROOT}/.venv/bin/python")
    else
        die "No environment found. Install uv (https://docs.astral.sh/uv/) then run scripts/setup.sh."
    fi
}

require_env() {
    resolve_python
    "${PY_RUN[@]}" -c 'import isaaclab' >/dev/null 2>&1 \
        || die "Isaac Lab is not importable in this environment. Run scripts/setup.sh first."
}

# Most recent checkpoint under a training log root, or empty if there is none.
latest_checkpoint() {
    local root="${1:-${EXT_ROOT}/logs/ppo_amp}"
    [[ -d "${root}" ]] || return 0
    find "${root}" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
}
