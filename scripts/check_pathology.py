# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac-Lab-backed checks that the pathological-gait machinery is actually live.

The analytical predecessor could only be checked against itself. These checks run the
real PhysX-backed environment and assert on quantities that only exist because the
physics is real: contact-derived support polygons, feed-forward reflex torque against
simulated joint state, and per-environment actuator randomization written into the
solver.

Exits non-zero if any check fails, so it can be used as a gate.

.. note::
    The simulation app is launched before any task import; see ``zero_agent.py`` for why.
"""

import argparse

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-H1-Pathological-Gait-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs only once the simulation app is up."""

import importlib

import os
import sys
import traceback
import gymnasium as gym
import torch

from isaaclab_tasks.utils import load_cfg_from_registry, resolve_presets

importlib.import_module("humanoid_pathological_gait.tasks")

from isaaclab.managers import SceneEntityCfg  # noqa: E402

from humanoid_pathological_gait.tasks.humanoid_pathological_gait.mdp.rewards import compute_xcom_and_mos  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    """Record one check outcome."""
    results.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def main() -> int:
    env_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg)
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    # Hold spasticity at full strength: the curriculum would otherwise start it at zero
    # and this run is far too short to ramp it in.
    env_cfg.curriculum.spasticity = None
    env_cfg.initial_spasticity_scale = 1.0

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]

    # -- joint ordering
    check(
        "joint layout covers all 19 H1 joints",
        len(env.joint_layout.sim_names) == 19 and robot.num_joints == 19,
        f"{robot.num_joints} joints",
    )

    # -- reference stride resets place the robot on the reference pose
    q_ref, _ = env.reference_gait.sample()
    reset_error = (robot.data.joint_pos.torch - q_ref).abs().max()
    check(
        "reset places joints on the reference pose",
        reset_error < 0.2,
        f"max |q - q_ref| = {reset_error:.4f} rad right after reset",
    )

    # -- both paretic sides are represented
    left_fraction = float((env.reference_gait.paretic_side < 0).float().mean())
    check(
        "paretic side is sampled on both sides",
        0.2 < left_fraction < 0.8,
        f"{left_fraction:.2%} of environments are left-paretic",
    )

    # -- asymmetric randomization reached the solver, tighter on the paretic limb
    is_left_paretic = env.reference_gait.paretic_side < 0
    stiffness_ratio = robot.data.joint_stiffness.torch / env.default_joint_stiffness.clamp(min=1e-6)
    left_dev = (stiffness_ratio[:, env.joint_layout.left_leg_ids] - 1.0).abs().amax(dim=-1)
    right_dev = (stiffness_ratio[:, env.joint_layout.right_leg_ids] - 1.0).abs().amax(dim=-1)
    paretic_dev = torch.where(is_left_paretic, left_dev, right_dev)
    sound_dev = torch.where(is_left_paretic, right_dev, left_dev)
    check(
        "paretic leg stiffness is held near nominal",
        float(paretic_dev.max()) < 0.05,
        f"max paretic deviation {float(paretic_dev.max()):.4f}",
    )
    check(
        "sound leg stiffness is randomized more widely than the paretic leg",
        float(sound_dev.mean()) > 3.0 * float(paretic_dev.mean()),
        f"mean sound {float(sound_dev.mean()):.4f} vs paretic {float(paretic_dev.mean()):.4f}",
    )

    effort_ratio = robot.data.joint_effort_limits.torch / env.default_joint_effort_limits.clamp(min=1e-6)
    paretic_effort = torch.where(
        is_left_paretic.unsqueeze(-1),
        effort_ratio[:, env.joint_layout.left_leg_ids],
        effort_ratio[:, env.joint_layout.right_leg_ids],
    )
    check(
        "paretic leg carries a reduced torque ceiling",
        float(paretic_effort.max()) < 0.6,
        f"max paretic effort-limit ratio {float(paretic_effort.max()):.3f}",
    )

    # -- roll out and watch the pathology-specific quantities
    foot_names = list(env.cfg.rewards.margin_of_stability.params["asset_cfg"].body_names)
    foot_asset_cfg = SceneEntityCfg("robot", body_names=foot_names, preserve_order=True)
    foot_asset_cfg.resolve(env.scene)
    foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=foot_names, preserve_order=True)
    foot_sensor_cfg.resolve(env.scene)

    spastic_peak = 0.0
    spastic_envs = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    contact_seen = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)
    mos_samples = []
    phase_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    sound_limb_leak = 0.0

    left_ids = env.joint_layout.left_leg_ids.unsqueeze(0).expand(env.num_envs, -1)
    right_ids = env.joint_layout.right_leg_ids.unsqueeze(0).expand(env.num_envs, -1)

    for _ in range(args_cli.num_steps):
        actions = torch.zeros(env.action_space.shape, device=env.device)
        env.step(actions)

        spastic_peak = max(spastic_peak, float(env.applied_spastic_torque.abs().max()))
        spastic_envs |= env.applied_spastic_torque.abs().amax(dim=-1) > 1.0

        # Read the paretic side inside the loop: environments reset mid-rollout and draw
        # a new side, so a mask captured before the loop would go stale.
        step_left_paretic = (env.reference_gait.paretic_side < 0).unsqueeze(-1)
        sound_ids = torch.where(step_left_paretic, right_ids, left_ids)
        sound_limb_leak = max(
            sound_limb_leak, float(torch.gather(env.applied_spastic_torque, 1, sound_ids).abs().max())
        )

        _, mos, in_contact = compute_xcom_and_mos(env, foot_asset_cfg, foot_sensor_cfg)
        contact_seen |= in_contact
        mos_samples.append(mos)
        phase_seen |= env.reference_gait.gait_phase > 0.5

    check(
        "TSRT reflex torque engages on the paretic limb",
        spastic_peak > 1.0 and bool(spastic_envs.any()),
        f"peak |tau_spastic| = {spastic_peak:.1f} Nm in {int(spastic_envs.sum())}/{env.num_envs} environments",
    )

    check(
        "reflex torque never reaches the sound limb",
        sound_limb_leak == 0.0,
        f"max {sound_limb_leak:.3e} Nm over the rollout",
    )

    check(
        "both feet register contact during the rollout",
        bool(contact_seen.all()),
        f"{int(contact_seen.sum())}/{contact_seen.numel()} foot-environment pairs made contact",
    )

    mos_all = torch.cat(mos_samples)
    check(
        "margin of stability is finite and physically scaled",
        bool(torch.isfinite(mos_all).all()) and float(mos_all.abs().max()) < 5.0,
        f"MoS range [{float(mos_all.min()):.3f}, {float(mos_all.max()):.3f}] m",
    )

    check("gait phase advances past mid-stride", bool(phase_seen.any()), f"{int(phase_seen.sum())} environments")

    env.close()

    failed = [name for name, passed, _ in results if not passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
    if failed:
        print("Failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    # Isaac registers atexit handlers that force a zero exit status, so an unhandled
    # exception here used to surface as SUCCESS: this script exited 0 while printing a
    # traceback. verify.sh gates on `$?` and train_and_evaluate_seeds.sh checks it too, so a
    # crashed run read as a passing one. Catch, close the app, flush, then bypass atexit.
    try:
        exit_code = main()
    except BaseException:  # noqa: BLE001 - the status must survive any failure, including SystemExit
        traceback.print_exc()
        exit_code = 1
    # Order matters: ``simulation_app.close()`` terminates the process itself with status 0,
    # so on the failure path it must not run -- otherwise the crash is reported as success.
    # os._exit is safe here; the OS reclaims everything the app was holding.
    sys.stdout.flush()
    sys.stderr.flush()
    if exit_code:
        os._exit(int(exit_code))
    simulation_app.close()
    os._exit(0)
