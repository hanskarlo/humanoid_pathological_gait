# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Find the H1 arm posture whose forearm tips reach the walker handle (walker plan §2.3).

The patient models' arms track the patient's captured arm swing, but a walker user's hands rest
on the handle. In walker mode the arm reference is held at a fixed grip posture; this measures
one, on the robot as it actually builds:

1. The forearm direction and length in each ``*_elbow_link`` frame, from the link's centre of mass
   (which lies along the forearm) -- the H1 has no hand link, so the forearm tip is the grip.
2. A grid over shoulder pitch and elbow angle (shoulder roll set for the lateral reach, yaw 0):
   for each posture, where the forearm tip lands in the pelvis frame. The best posture puts the
   tips at the target: ``handle_ahead`` forward, handle height, symmetric.

The robot is held by writing joint states and stepping once per posture with gravity off, so the
measurement is kinematic.

    .venv/bin/python scripts/measure_grip_posture.py --headless
"""

import argparse

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--handle_height", type=float, default=0.933, help="Handle above ground, m (asset).")
parser.add_argument("--probe", action="store_true", help="Print a few hand-picked postures and exit.")
parser.add_argument("--handle_ahead", type=float, default=0.35, help="Target forward reach from the pelvis, m.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs only once the simulation app is up."""

import importlib
import itertools
import math
import os
import sys

import gymnasium as gym
import torch

from isaaclab_tasks.utils import load_cfg_from_registry

importlib.import_module("humanoid_pathological_gait.tasks")
TASK = "Isaac-H1-Pathological-Gait-Play-v0"


def to_body(quat_xyzw: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
    q = -quat_xyzw[..., :3]
    w = quat_xyzw[..., 3:4]
    t = 2.0 * torch.cross(q, vec_w, dim=-1)
    return vec_w + w * t + torch.cross(q, t, dim=-1)


def to_world(quat_xyzw: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
    q = quat_xyzw[..., :3]
    w = quat_xyzw[..., 3:4]
    t = 2.0 * torch.cross(q, vec_b, dim=-1)
    return vec_b + w * t + torch.cross(q, t, dim=-1)


def main() -> int:
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.gravity = (0.0, 0.0, 0.0)
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    names = robot.joint_names
    elbows = [robot.body_names.index(n) for n in ("left_elbow_link", "right_elbow_link")]

    com_b = robot.data.body_com_pos_b.torch[0, elbows]
    print(f"[forearm] elbow-link CoM in link frame (left, right): {com_b.tolist()}")
    # Tip = twice the CoM offset along the same direction: a forearm of roughly uniform mass.
    tip_b = 2.0 * com_b
    print(f"[forearm] estimated tip offset (left, right): {[[round(x, 3) for x in t] for t in tip_b.tolist()]}")
    print(f"[forearm] estimated forearm length: {torch.linalg.norm(tip_b, dim=-1).tolist()}")

    q0 = robot.data.joint_pos.torch.clone()
    arm = {
        k: (names.index(f"left_{k}"), names.index(f"right_{k}"))
        for k in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
    }
    limits = {k: robot.data.soft_joint_pos_limits.torch[0, i].tolist() for k, (i, _) in arm.items()}
    print(f"[limits] arm joint limits: {limits}")

    def tip_for(pitch, elbow, roll):
        q = q0.clone()
        for key, value, mirror in (
            ("shoulder_pitch", pitch, 1.0),
            ("shoulder_roll", roll, -1.0),
            ("shoulder_yaw", 0.0, -1.0),
            ("elbow", elbow, 1.0),
        ):
            li, ri = arm[key]
            q[0, li], q[0, ri] = value, mirror * value
        for _ in range(3):
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.set_joint_position_target(q)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.sim.get_physics_dt())
        pelvis_p = robot.data.root_link_pos_w.torch[0]
        pelvis_q = robot.data.root_link_quat_w.torch[0]
        link_p = robot.data.body_link_pos_w.torch[0, elbows]
        link_q = robot.data.body_link_quat_w.torch[0, elbows]
        tip_w = link_p + to_world(link_q, tip_b)
        achieved = robot.data.joint_pos.torch[0, list(arm["elbow"]) + list(arm["shoulder_pitch"])]
        return (
            to_body(pelvis_q.expand(2, 4), link_p - pelvis_p),
            to_body(pelvis_q.expand(2, 4), tip_w - pelvis_p),
            pelvis_p,
            achieved,
        )

    if args_cli.probe:
        for posture in ((0.0, 0.0, 0.0), (0.0, -1.0, 0.0), (-0.5, -1.0, 0.0), (0.5, -1.0, 0.0), (0.0, 1.5, 0.0)):
            elbow_rel, tip_rel, pelvis_p, achieved = tip_for(*posture)
            got = [round(x, 3) for x in achieved.tolist()]
            elbow_at = [round(x, 3) for x in elbow_rel[0].tolist()]
            tip_at = [round(x, 3) for x in tip_rel[0].tolist()]
            print(
                f"[probe] pitch {posture[0]:+.2f} elbow {posture[1]:+.2f} roll {posture[2]:+.2f}"
                f" | achieved elbow/pitch {got} | pelvis z {pelvis_p[2].item():.3f}"
                f" | left elbow joint (fwd,left,up) {elbow_at} | left tip {tip_at}"
            )
        env.close()
        return 0

    target_up = args_cli.handle_height  # world z of the handle
    best = []
    # H1 conventions measured by --probe: elbow q = 0 is the forearm horizontal and forward and
    # POSITIVE q extends it downward (+1.57 ~ straight); positive shoulder pitch swings the arm back.
    for pitch, elbow, roll in itertools.product(
        [x * 0.05 for x in range(-14, 3)], [x * 0.05 for x in range(18, 34)], [-0.1, 0.0, 0.1]
    ):
        _, tip_rel, pelvis_p, _ = tip_for(pitch, elbow, roll)
        fwd = tip_rel[:, 0].mean().item()
        height = pelvis_p[2].item() + tip_rel[:, 2].mean().item()
        lateral = (tip_rel[0, 1] - tip_rel[1, 1]).item() / 2.0
        err = math.hypot(fwd - args_cli.handle_ahead, height - target_up)
        best.append((err, pitch, elbow, roll, fwd, lateral, height))
    best.sort()
    print(f"[sweep] target: forward {args_cli.handle_ahead} m from pelvis, tip height {target_up} m")
    print("[sweep] best postures (err m, shoulder_pitch, elbow, shoulder_roll | fwd, half-separation, height m):")
    for row in best[:8]:
        print("   " + "  ".join(f"{x:+.3f}" for x in row))
    env.close()
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        sys.stdout.flush()
    os._exit(code)
