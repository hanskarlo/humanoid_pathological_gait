# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 4a smoke: does the implicit grip build, clone to every environment correctly, and hold?

1. **Cloning.** Every environment's two grip joints exist, and each targets that environment's
   own elbow link and handle. A joint copied with its targets un-remapped would tie environment i's
   hands to environment 0's walker, metres away.
2. **Holding.** With a zero residual (the robot's PD tracks the reference), the hand-to-handle
   error per environment, finite forces, and the walker following its command. A zero-action robot
   falls within a second or two, so this checks the coupling, not a gait. The same numbers with
   ``--grip explicit`` are the baseline.

    .venv/bin/python scripts/check_walker_grip.py --headless [--grip explicit] [--stiffness 50000]

Import order: AppLauncher first (see zero_agent.py).
"""

import argparse

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=50, help="Control steps (20 ms each).")
parser.add_argument("--grip", choices=("explicit", "implicit"), default="implicit")
parser.add_argument("--stiffness", type=float, default=None, help="N/m; the grip model's default if unset.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs only once the simulation app is up."""

import importlib
import sys

import gymnasium as gym
import torch

from isaaclab.sim.utils import get_current_stage

from isaaclab_tasks.utils import load_cfg_from_registry

importlib.import_module("humanoid_pathological_gait.tasks")
from humanoid_pathological_gait.tasks.humanoid_pathological_gait.config.h1_pathological.h1_pathological_env_cfg import (  # noqa: E402
    apply_walker,
)

TASK = "Isaac-H1-Pathological-Gait-Play-v0"


def check_cloning(num_envs: int) -> int:
    """Count environments whose grip joints are missing or target another environment's bodies."""
    from pxr import UsdPhysics

    stage = get_current_stage()
    bad = 0
    for i in range(num_envs):
        env = f"/World/envs/env_{i}"
        for side in ("left", "right"):
            prim = stage.GetPrimAtPath(f"{env}/WalkerGrip/{side}_grip")
            if not prim.IsValid():
                print(f"[clone] env {i} {side}: joint MISSING")
                bad += 1
                continue
            joint = UsdPhysics.Joint(prim)
            b0, b1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            ok = len(b0) == 1 and len(b1) == 1 and all(str(t).startswith(env + "/") for t in (*b0, *b1))
            if i < 2 or not ok:
                verdict = "ok" if ok else "WRONG"
                print(f"[clone] env {i} {side}: body0 {[str(t) for t in b0]} body1 {[str(t) for t in b1]} {verdict}")
            bad += 0 if ok else 1
    print(f"[clone] {num_envs} environments, {bad} joints missing or mis-targeted")
    return bad


def report_reset_mismatch(env) -> None:
    """Grip point minus hand point right after reset, in the walker frame (x forward, y left, z up)."""
    term = env.action_manager.get_term("walker")
    p_hand, _, p_grip, _, root_quat = term._points()
    q = -root_quat[:, :3].unsqueeze(1).expand(-1, 2, -1)
    w = root_quat[:, 3:4].unsqueeze(1)
    d = p_grip - p_hand
    t = 2.0 * torch.cross(q, d, dim=-1)
    d_b = (d + w * t + torch.cross(q, t, dim=-1)) * 100
    for side, k in (("left", 0), ("right", 1)):
        m, sd = d_b[:, k].mean(0), d_b[:, k].std(0)
        mean, spread = [round(x, 2) for x in m.tolist()], [round(x, 2) for x in sd.tolist()]
        print(f"[reset] {side} hand: grip - hand in walker frame (fwd, left, up) cm: mean {mean} sd {spread}")


def main() -> int:
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    applied = apply_walker(env_cfg, args_cli.stiffness, grip=args_cli.grip)
    print(f"[walker] grip {applied['grip']}, K {applied['stiffness']:.0f} N/m, C {applied['damping']:.1f} N s/m")
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    bad = check_cloning(env.num_envs) if args_cli.grip == "implicit" else 0
    env.reset()
    report_reset_mismatch(env)

    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    errors, finite = [], True
    walker = env.scene["walker"]
    for step in range(args_cli.steps):
        env.step(actions)
        s = env.walker_state
        errors.append(s["spring_error"].clone())
        finite &= all(torch.isfinite(s[k]).all().item() for k in ("hand_force_w", "sensor_wrench", "cmd"))
        if step in (0, 1, 4, 9, 24, 49):
            print(
                f"[hold] step {step:3d}: hand-handle error mean {s['spring_error'].mean().item() * 100:5.2f} cm"
                f" max {s['spring_error'].max().item() * 100:5.2f} cm, grip force mean"
                f" {torch.linalg.norm(s['hand_force_w'], dim=-1).mean().item():7.1f} N,"
                f" v_cmd {s['cmd'][:, 0].mean().item():+.3f}"
            )
    e = torch.stack(errors) * 100  # (T, N, 2) cm
    per_env_max = e.amax(dim=(0, 2))
    offset = walker.data.root_link_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    print(f"[hold] per-environment max error, cm: {[round(x, 2) for x in per_env_max.tolist()]}")
    p99 = torch.quantile(e.flatten(), 0.99).item()
    print(f"[hold] error over all steps: median {e.median().item():.2f} cm, p99 {p99:.2f} cm")
    print(f"[hold] walker distance from its own env origin, m: {[round(x, 2) for x in offset.norm(dim=-1).tolist()]}")
    print(f"[hold] finite throughout: {finite}")
    env.close()
    return 0 if finite and bad == 0 else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        sys.stdout.flush()
    import os

    os._exit(code)
