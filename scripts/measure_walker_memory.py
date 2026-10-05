# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Host and GPU memory of the coupled (H1 + walker) scene at a given environment count.

The first walker fine-tune at 14,720 environments was OOM-killed on the host at 28.2 GB resident
while building the scene (2026-10-02). This measures the scene's footprint at smaller counts so the
cost per environment, and so the 14,720 figure, can be extrapolated before launching anything that
size again.

    .venv/bin/python scripts/measure_walker_memory.py --num_envs 512 --headless [--no_walker]
"""

import argparse

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--no_walker", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs only once the simulation app is up."""

import importlib
import os
import resource
import subprocess
import sys

import gymnasium as gym
import torch

from isaaclab_tasks.utils import load_cfg_from_registry

importlib.import_module("humanoid_pathological_gait.tasks")
from humanoid_pathological_gait.tasks.humanoid_pathological_gait.config.h1_pathological.h1_pathological_env_cfg import (  # noqa: E402
    apply_walker,
)

TASK = "Isaac-H1-Pathological-Gait-v0"


def rss_gb() -> float:
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024**2
    return float("nan")


def gpu_gb() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True
    )
    return float(out.stdout.strip().splitlines()[0]) / 1024


def main() -> int:
    before = rss_gb()
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if not args_cli.no_walker:
        apply_walker(env_cfg)
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(5):
        env.step(actions)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(
        f"[memory] envs {args_cli.num_envs} walker {not args_cli.no_walker}: host RSS {rss_gb():.2f} GB "
        f"(peak {peak:.2f}, at start {before:.2f}), GPU used {gpu_gb():.2f} GB",
        flush=True,
    )
    env.close()
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        sys.stdout.flush()
    os._exit(code)
