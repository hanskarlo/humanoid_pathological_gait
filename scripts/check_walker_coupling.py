# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""First run of the coupled scene: measure the placeholders, then check the coupling holds together.

Walker plan step 2-3 smoke (``docs/walker_evaluation_plan.md``), run before validations V1-V7:

1. **Measure** the constants ``WalkerCouplingActionCfg`` marks MEASURE, on the robot as it actually
   resets: its body names, and where each ``*_elbow_link`` sits relative to the pelvis in the
   pelvis frame (forward reach, separation, height). Those set ``hand_offset_body``,
   ``half_grip_width`` and the reset event's ``handle_ahead``.
2. **Smoke** the coupled scene with a zero residual (the robot's PD tracks the reference): grip
   forces finite, the spring error bounded, the walker base following its command, the sensor
   wrench and halt flags populated. A zero-action robot falls within a second or two; this checks
   the plumbing, not the gait.

    .venv/bin/python scripts/check_walker_coupling.py --headless

Import order: AppLauncher first (see zero_agent.py).
"""

import argparse

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=50, help="Control steps (20 ms each) for the smoke phase.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs only once the simulation app is up."""

import importlib
import sys

import gymnasium as gym
import torch

from isaaclab_tasks.utils import load_cfg_from_registry

importlib.import_module("humanoid_pathological_gait.tasks")
from humanoid_pathological_gait.tasks.humanoid_pathological_gait.config.h1_pathological.h1_pathological_env_cfg import (  # noqa: E402
    apply_walker,
)

TASK = "Isaac-H1-Pathological-Gait-Play-v0"


def to_body(quat_xyzw: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
    q = -quat_xyzw[:, :3]
    w = quat_xyzw[:, 3:4]
    t = 2.0 * torch.cross(q, vec_w, dim=-1)
    return vec_w + w * t + torch.cross(q, t, dim=-1)


def main() -> int:
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    applied = apply_walker(env_cfg)
    print(f"[walker] applied: {applied}")
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]

    # -- 1. measure ---------------------------------------------------------------------------
    print(f"[measure] robot bodies: {robot.body_names}")
    arm = [n for n in robot.body_names if any(k in n for k in ("elbow", "hand", "wrist", "shoulder"))]
    print(f"[measure] arm bodies: {arm}")
    pelvis_pos = robot.data.root_link_pos_w.torch
    pelvis_quat = robot.data.root_link_quat_w.torch
    for name in arm:
        idx = robot.body_names.index(name)
        rel = to_body(pelvis_quat, robot.data.body_link_pos_w.torch[:, idx] - pelvis_pos)
        print(f"[measure] {name:24s} in pelvis frame (fwd, left, up) m: {[round(x, 3) for x in rel.mean(0).tolist()]}")
    print(f"[measure] pelvis height m: {pelvis_pos[:, 2].mean().item():.3f}")

    # -- 2. smoke -------------------------------------------------------------------------------
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    worst_error, worst_force, finite = 0.0, 0.0, True
    for _ in range(args_cli.steps):
        env.step(actions)
        s = env.walker_state
        finite &= all(torch.isfinite(s[k]).all().item() for k in ("hand_force_w", "sensor_wrench", "cmd"))
        worst_error = max(worst_error, s["spring_error"].max().item())
        worst_force = max(worst_force, torch.linalg.norm(s["hand_force_w"], dim=-1).max().item())
    s = env.walker_state
    print(f"[smoke] finite throughout: {finite}")
    print(f"[smoke] max spring error {worst_error * 100:.2f} cm, max grip force {worst_force:.1f} N")
    print(f"[smoke] last sensor wrench (Fx,Fy,Fz,Tx,Ty,Tz): {[round(x, 2) for x in s['sensor_wrench'][0].tolist()]}")
    print(f"[smoke] last command (v_x, omega_z): {[round(x, 3) for x in s['cmd'][0].tolist()]}")
    print(f"[smoke] halt flags (deadman, collapse, impulse) env 0: {s['halt'][0].tolist()}")
    env.close()
    return 0 if finite else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        sys.stdout.flush()
    import os

    os._exit(code)
