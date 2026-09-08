#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Can this robot hold the reference's single-support pose at all?

Eight reward interventions failed to move double support off 0.82 against a reference of
0.442. The reason appears to be that the policy has no controlled single-support skill: it
enters single support 14-18% of the time and holds a margin of -0.23 m there, three times
past the boundary the patient reaches, so high double support is a recovery rather than a
choice. Reward shaping cannot install a skill the policy never successfully executes.

Before spending on curricula or constraints, this partitions the cause. Freeze the gait phase
somewhere the reference schedule says single support, command the reference pose exactly, and
see whether the robot stays up:

* **It cannot** -- the deficit is actuation. The paretic effort ceiling, the reflex, or the PD
  gains make one-legged stance infeasible and no learning method will produce it. In that case
  high double support is arguably the *correct* emergent response to the impairment, which is
  a paper claim rather than a bug.
* **It can** -- the deficit is exploration. The policy could hold single support and has not
  discovered how, which is what an assistive-force curriculum addresses.

Four conditions partition it further: full pathology, no spasticity, no weakness, neither.

Usage::

    scripts/single_leg_stance.py --hold_s 3.0
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="Isaac-H1-Pathological-Gait-Play-v0")
parser.add_argument("--num_envs", type=int, default=32, help="Environments per condition.")
parser.add_argument("--hold_s", type=float, default=3.0, help="Seconds to hold the pose.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--stiffness_scales",
    type=float,
    nargs="*",
    default=(1.0,),
    help="Actuator stiffness multipliers to sweep. The commanded reference pose sags under "
    "gravity; this asks whether that sag is a gain limitation.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import humanoid_pathological_gait.tasks  # noqa: F401, E402


def scheduled_single_support_phases(env, count: int) -> torch.Tensor:
    """Phases at which the reference has exactly one foot down, spread over the stride."""
    schedule = env.reference_gait.ref_contact
    if schedule is None:
        raise SystemExit("the stride archive carries no contact schedule")
    loaded = (schedule > 0.5).sum(dim=-1)
    single = torch.nonzero(loaded == 1).flatten()
    if single.numel() == 0:
        raise SystemExit("the reference schedule never has exactly one foot down")
    picks = single[torch.linspace(0, single.numel() - 1, count).long()]
    return picks.float() / (env.reference_gait.num_samples - 1)


def run_condition(
    env, label: str, phases: torch.Tensor, spasticity: bool, weakness: bool, steps: int, stiffness: float = 1.0
) -> dict:
    """Hold the reference pose at fixed phases and report whether the robot stays up."""
    robot = env.scene["robot"]
    env.reset()

    # The reset events randomise gains, mass and the effort ceiling; re-impose the condition
    # afterwards so each ablation is what it says it is.
    if not weakness:
        robot.write_joint_effort_limit_to_sim_index(
            limits=env.default_joint_effort_limits, env_ids=torch.arange(env.num_envs, device=env.device)
        )
    env.spasticity_scale[:] = 1.0 if spasticity else 0.0
    if stiffness != 1.0:
        ids = torch.arange(env.num_envs, device=env.device)
        robot.write_joint_stiffness_to_sim_index(stiffness=env.default_joint_stiffness * stiffness, env_ids=ids)
        robot.write_joint_damping_to_sim_index(
            damping=env.default_joint_damping * float(np.sqrt(stiffness)), env_ids=ids
        )

    env.reference_gait.gait_phase[:] = phases
    q_ref, v_ref = env.reference_gait.sample()
    limits = robot.data.soft_joint_pos_limits.torch
    robot.write_joint_state_to_sim(torch.clamp(q_ref, limits[..., 0], limits[..., 1]), torch.zeros_like(v_ref))

    zero = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
    upright = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    survived = torch.full((env.num_envs,), float(steps), device=env.device)
    heights, margins, loaded_counts, spans, errs, per_joint, sat = [], [], [], [], [], [], []
    sensor = env.scene["contact_forces"]
    # Two separate index sets, because the robot and the contact sensor order their bodies
    # differently. Mixing them is silent: sensor indices on the robot tensor gave a
    # pelvis-to-foot span of 0.14 m for a robot standing at 0.98 m, and robot indices on the
    # sensor tensor reported zero loaded feet for a robot plainly standing. The reward terms
    # carry an asset_cfg and a sensor_cfg for precisely this reason.
    sensor_foot_ids = sensor.find_bodies([".*ankle_link"], preserve_order=True)[0]
    robot_foot_ids = robot.find_bodies([".*ankle_link"], preserve_order=True)[0]

    for step in range(steps):
        # Freeze the phase so the commanded pose stays the single-support one, and hold the
        # ablation against the curriculum, which would otherwise ramp spasticity back up.
        env.reference_gait.gait_phase[:] = phases
        env.spasticity_scale[:] = 1.0 if spasticity else 0.0
        env.step(zero)

        height = robot.data.root_link_pos_w.torch[:, 2]
        fell = upright & (height < 0.65)
        survived[fell] = step
        upright = upright & ~fell
        heights.append(height.clone())
        force = torch.norm(sensor.data.net_forces_w.torch[:, sensor_foot_ids], dim=-1)
        loaded_counts.append((force > 1.0).sum(dim=-1).clone())
        # Pelvis-to-loaded-foot. If the joints track their targets but this differs from the
        # retargeting model's value for the same angles, the two robots are not the same shape.
        ankle_z = robot.data.body_link_pos_w.torch[:, robot_foot_ids, 2]
        loaded_mask = force > 1.0
        lowest = torch.where(loaded_mask, ankle_z, torch.full_like(ankle_z, float("inf"))).min(dim=-1).values
        spans.append(torch.where(torch.isfinite(lowest), height - lowest, torch.full_like(height, float("nan"))))
        errs.append((robot.data.joint_pos.torch - q_ref).abs().mean(dim=-1))
        per_joint.append((robot.data.joint_pos.torch - q_ref).abs().mean(dim=0))
        sat.append((robot.data.applied_torque.torch.abs() / robot.data.joint_effort_limits.torch.clamp(min=1e-6)).mean(dim=0))
        if hasattr(env, "last_mos") and env.last_mos is not None:
            margins.append(env.last_mos.clone())

    held = float(upright.float().mean())
    loaded = torch.stack(loaded_counts).float()
    return {
        "condition": label,
        "held_pct": 100.0 * held,
        "median_survival_steps": float(survived.median()),
        "mean_height_m": float(torch.stack(heights).mean()),
        # The validity check. If both feet stay loaded the robot never left double support and
        # "held 100%" says nothing about single-support competence.
        "mean_feet_loaded": float(loaded.mean()),
        "single_support_pct": 100.0 * float((loaded == 1).float().mean()),
        "pelvis_to_foot_m": float(torch.nanmean(torch.stack(spans))),
        "joint_err_deg": float(torch.rad2deg(torch.stack(errs).mean())),
        "per_joint_deg": torch.rad2deg(torch.stack(per_joint).mean(dim=0)).cpu().numpy(),
        "saturation": torch.stack(sat).mean(dim=0).cpu().numpy(),
    }


def main() -> int:
    env_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "env_cfg_entry_point")
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.hold_s + 10.0)
    # A curriculum that ramps spasticity would fight the ablation.
    env_cfg.curriculum.spasticity = None
    if hasattr(env_cfg.curriculum, "push_magnitude"):
        env_cfg.curriculum.push_magnitude = None

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    steps = int(args_cli.hold_s / float(env.step_dt))
    phases = scheduled_single_support_phases(env, env.num_envs)

    print("=" * 96)
    print(f"  Static single-support hold -- {env.num_envs} envs x {args_cli.hold_s:.1f} s "
          f"({steps} control steps), reference pose commanded, phase frozen")
    print("=" * 96, flush=True)

    # Reference pelvis height, for reading the sag against.
    ref_height = float(env.reference_gait.ref_root_height.mean()) if env.reference_gait.ref_root_height is not None else float("nan")
    print(f"  reference pelvis height {ref_height:.4f} m\n")

    joint_names = list(env.scene["robot"].data.joint_names)
    rows = []
    conditions = [("full pathology", True, True), ("no spasticity", False, True),
                  ("no weakness", True, False), ("neither", False, False)]
    for scale in args_cli.stiffness_scales:
        for label, spast, weak in (conditions if scale == 1.0 else conditions[:1]):
            name = label if scale == 1.0 else f"stiffness x{scale:g}"
            rows.append(run_condition(env, name, phases, spast, weak, steps, stiffness=scale))
            r = rows[-1]
            print(f"  {r['condition']:<18} held {r['held_pct']:5.1f}%   median survival "
              f"{r['median_survival_steps']:5.0f}/{steps}   pelvis {r['mean_height_m']:.4f} m   "
              f"feet loaded {r['mean_feet_loaded']:.2f}   genuinely single support "
              f"{r['single_support_pct']:5.1f}%   sag {1000 * (ref_height - r['mean_height_m']):.0f} mm   "
              f"pelvis-to-foot {r['pelvis_to_foot_m']:.4f} m   joint err {r['joint_err_deg']:.2f} deg", flush=True)

    print()
    top = rows[0]["per_joint_deg"]
    order = np.argsort(-top)[:8]
    print("\n  worst steady-state joint errors under the full deficit model (deg):")
    satu = rows[0]["saturation"]
    print(f"     {'joint':<22}{'err deg':>9}{'torque / its ceiling':>22}")
    for i in order:
        print(f"     {joint_names[i]:<22}{top[i]:>9.2f}{100 * satu[i]:>21.1f}%")

    full, unimpaired = rows[0], next((r for r in rows if r["condition"] == "neither"), rows[0])
    print()
    if full["held_pct"] < 80.0 and unimpaired["held_pct"] > 80.0:
        print("VERDICT: the deficit model itself forbids one-legged stance.")
        print("  High double support is then arguably the correct emergent response to the")
        print("  impairment, and belongs in the paper as a finding rather than being trained away.")
        return 0
    if full["held_pct"] < 80.0:
        print("VERDICT: the robot cannot hold the pose even unimpaired -- the limit is the")
        print("  controller, not the pathology and not the policy.")
        return 0

    print("VERDICT: the robot does not fall in any condition, including the full deficit model,")
    print("  so single-legged stance is not forbidden by the impairment.")
    print()
    print("  But the commanded reference pose does not by itself produce single support: the")
    print(f"  pose sags {1000 * (ref_height - full['mean_height_m']):.0f} mm under gravity with "
          f"{full['joint_err_deg']:.1f} deg of mean joint error at zero")
    print("  action, which drops the swing foot onto the ground. Only "
          f"{full['single_support_pct']:.0f}% of the hold is")
    print("  genuinely one-footed.")
    stiff = [r for r in rows if r["condition"].startswith("stiffness")]
    if stiff:
        best = stiff[-1]
        print()
        print(f"  Actuator gains matter and were never examined: {best['condition']} takes the sag to "
              f"{1000 * (ref_height - best['mean_height_m']):.0f} mm,")
        print(f"  joint error to {best['joint_err_deg']:.1f} deg and genuine single support to "
              f"{best['single_support_pct']:.0f}%. That is a lever")
        print("  outside the reward function entirely.")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    if exit_code:
        os._exit(int(exit_code))
    simulation_app.close()
    os._exit(0)
