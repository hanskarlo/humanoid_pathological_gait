#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# One-time environment setup: resolve dependencies, accept the Omniverse EULA, warm the
# Isaac Sim asset cache, and report whether the clinical data files are staged.
#
# Usage: scripts/setup.sh [--skip-sync]

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

SKIP_SYNC=0
for arg in "$@"; do
    case "${arg}" in
        --skip-sync) SKIP_SYNC=1 ;;
        -h|--help) sed -n '4,9p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) die "Unknown argument: ${arg}" ;;
    esac
done

info "Extension root: ${EXT_ROOT}"

if [[ "${SKIP_SYNC}" -eq 0 ]]; then
    command -v uv >/dev/null 2>&1 || die "uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"
    info "Resolving the project environment (this downloads ~25 GB of Isaac Sim on a cold cache)"
    (cd "${EXT_ROOT}" && uv sync)
    ok "environment synced"
fi

resolve_python
ensure_eula

info "Checking Isaac Lab and GPU availability"
"${PY_RUN[@]}" - <<'PY'
import sys

import isaaclab
import torch

print(f"  isaaclab       {isaaclab.__version__}")
print(f"  torch          {torch.__version__}")
if torch.cuda.is_available():
    print(f"  cuda device    {torch.cuda.get_device_name(0)}")
else:
    print("  cuda device    NONE -- training needs an NVIDIA GPU", file=sys.stderr)
    sys.exit(1)
PY
ok "Isaac Lab importable, GPU visible"

info "Checking staged clinical data"
"${PY_RUN[@]}" - <<'PY'
from humanoid_pathological_gait.tasks.humanoid_pathological_gait import assets

try:
    print(f"  reference stride  {assets.reference_stride_path()}")
except FileNotFoundError as exc:
    raise SystemExit(f"  MISSING reference stride -- the task cannot run without it.\n{exc}")

try:
    print(f"  AMP expert corpus {assets.expert_dataset_path()}")
except FileNotFoundError:
    print("  AMP expert corpus NOT STAGED -- training falls back to a weaker synthetic prior.")
    print("                    See docs/data.md to stage the real corpus.")
PY

info "Booting Isaac Sim headless to warm the kernel and asset caches (first run takes several minutes)"
"${PY_RUN[@]}" "${SCRIPT_DIR}/zero_agent.py" --task Isaac-H1-Pathological-Gait-v0 \
    --num_envs 4 --num_steps 5 --headless >/dev/null 2>&1 \
    || die "Isaac Sim failed to boot. Run scripts/verify.sh for the full output."
ok "Isaac Sim boots and the H1 asset resolves"

printf '\n%sSetup complete.%s Next:\n' "${C_BOLD}" "${C_OFF}"
printf '  scripts/verify.sh              # full preflight checks\n'
printf '  scripts/train.sh --preset smoke # 20-iteration smoke test\n'
printf '  scripts/train.sh               # full training run\n'
