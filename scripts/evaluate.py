# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture a full evaluation rollout of a trained checkpoint, for the paper's figures.

``play.py`` prints four summary numbers. This writes the underlying per-step tensors to
disk, so every figure and table in the paper is generated from one archived rollout
rather than from a separate simulation run per figure -- which is what makes the numbers
in the text and the curves in the figures agree.

What it captures that the superseded analytical evaluator could not:

* **Real support polygons.** The margin of stability comes from the environment's own
  ``compute_xcom_and_mos``: whole-body centre of mass, and a support polygon built from
  the feet whose contact sensors are actually loaded. The analytical evaluator placed
  synthetic feet at +/- 0.15 m from the root and assumed permanent double support, so its
  MoS was a function of the root pose alone.
* **Contact-derived gait events.** Stance and swing come from foot contact forces, so
  step time, stance fraction and temporal asymmetry are measured, not inferred from a
  forward-kinematics height threshold.
* **A deterministic, balanced paretic-side split.** Exactly half the environments are
  left-paretic and half right-paretic, fixed for the whole rollout, so per-limb averages
  are not contaminated by an unbalanced draw.

Every environment is mirrored into a common **paretic/sound** frame before averaging,
using the layout's own tested mirror map. Averaging left- and right-paretic environments
without that step cancels exactly the asymmetry the paper is about.

Outputs, under ``--output_dir``:

===========================  ==========================================================
``rollout.npz``              Per-step tensors, both raw and paretic/sound standardized.
``gait_cycle.npz``           Cycle-normalized (0-100%) mean and SD per joint per limb.
``metrics.json``             Every scalar metric, machine-readable.
``metrics.csv``              The same metrics as a table, for the paper's results table.
===========================  ==========================================================

.. note::
    The simulation app is launched before any task import; see ``zero_agent.py`` for why.
"""

import argparse

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a train_amp.py checkpoint (.pt).")
parser.add_argument(
    "--zero_actions",
    action="store_true",
    help="Skip the checkpoint and drive zero actions, which commands the reference pose exactly. "
    "This measures what the reference stride itself scores on every gait metric -- the ceiling any "
    "policy tracking it can reach.",
)
parser.add_argument("--task", type=str, default="Isaac-H1-Pathological-Gait-Play-v0")
parser.add_argument("--num_envs", type=int, default=64, help="Environments; rounded down to an even number.")
parser.add_argument("--num_steps", type=int, default=1000, help="Control steps to record after warmup.")
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=100,
    help="Steps to discard before recording, letting the reset transient settle.",
)
parser.add_argument("--output_dir", type=str, default=None, help="Defaults to <checkpoint dir>/evaluation.")
parser.add_argument("--label", type=str, default=None, help="Name for this condition in the metrics table.")
parser.add_argument("--seed", type=int, default=0, help="Seed for the environment.")
parser.add_argument(
    "--stochastic", action="store_true", help="Sample actions instead of taking the policy mean (not for the paper)."
)
parser.add_argument(
    "--presets", type=str, nargs="*", default=(), help="Preset variants to select, e.g. --presets newton_mjwarp."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs only once the simulation app is up."""

import csv
import importlib
import json
from datetime import datetime
from pathlib import Path

import os
import sys
import traceback
import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import load_cfg_from_registry, resolve_presets

importlib.import_module("humanoid_pathological_gait.tasks")

from gait_analysis import (  # noqa: E402
    NUM_CYCLE_BINS,
    contact_gait_metrics,
    cycle_normalize,
    pelvic_obliquity,
    standardize_to_paretic_frame,
)

from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # noqa: E402

from humanoid_pathological_gait.algorithms.ppo import ActorCritic  # noqa: E402
from humanoid_pathological_gait.tasks.humanoid_pathological_gait.h1_joints import (  # noqa: E402
    CLINICAL_JOINT_ORDER,
    FOOT_BODY_NAMES,
)
from humanoid_pathological_gait.tasks.humanoid_pathological_gait.mdp.rewards import compute_xcom_and_mos  # noqa: E402

GRAVITY = 9.81


class NoSurvivingEnvironments(RuntimeError):
    """Raised when every environment fell, leaving no gait to summarize."""


class RolloutRecorder:
    """Accumulates per-step tensors on the CPU and stacks them at the end.

    Recording lands on the CPU each step rather than growing GPU buffers: the whole point
    of this script is to run alongside whatever else is using the card, and a 1000-step
    rollout of 64 environments is only a few tens of megabytes in host memory.
    """

    def __init__(self):
        self._buffers: dict[str, list[np.ndarray]] = {}

    def add(self, **tensors: torch.Tensor) -> None:
        """Record one step's worth of named tensors."""
        for name, tensor in tensors.items():
            self._buffers.setdefault(name, []).append(tensor.detach().cpu().numpy().copy())

    def stack(self) -> dict[str, np.ndarray]:
        """Stack every buffer along a leading time axis."""
        return {name: np.stack(frames, axis=0) for name, frames in self._buffers.items()}


def assign_balanced_paretic_sides(env, num_envs: int) -> torch.Tensor:
    """Fix the first half of the environments left-paretic and the second half right-paretic.

    The play config still draws the paretic side at random on every reset. That is fine
    for a demo and wrong for a measurement: a rollout with, say, 40 right-paretic and 24
    left-paretic environments weights the two mirror images unequally, and an environment
    that flips side mid-rollout contributes half a stride to each. Pinning the assignment
    and disabling the redraw makes each environment a stable, reproducible subject.
    """
    side = torch.ones(num_envs, device=env.device)
    side[: num_envs // 2] = -1.0  # -1 is left-paretic
    env.reference_gait.paretic_side[:] = side
    if env.cfg.events.reset_to_reference is not None:
        env.cfg.events.reset_to_reference.params["randomize_paretic_side"] = False
        env.event_manager.get_term_cfg("reset_to_reference").params["randomize_paretic_side"] = False
    return side


def assign_phase_offsets(env, num_envs: int) -> None:
    """Give every environment a distinct, deterministic starting phase.

    The Play config phase-locks every environment to the same trace (fixed
    ``start_phase``, no randomization), which a demo wants and a measurement cannot use.
    When ``stride_duration_s / dt`` is an exact integer the gait phase only ever takes
    that many distinct values, which round to that many of the 101 cycle bins; no number
    of recorded steps changes it, because every environment retraces the identical orbit.
    The 2026-09-04 baseline hit exactly this: a 1.2 s stride at 20 ms gave 60 reachable
    bins and 41 unreachable ones. The reference now carries the stride's own measured
    duration, so the ratio is 91.5 rather than 60 and the degenerate case does not apply
    at present -- but it returns for any reference whose duration is a round multiple of
    the control step, so the offsets below are kept rather than made conditional.

    Spreading a deterministic offset ``i / num_envs`` across environments turns that one
    orbit into ``num_envs`` phase-shifted copies of it, and their union fills in the gaps
    -- with enough environments, densely. It stays fully reproducible: the offsets are a
    function of environment count alone, not of any random draw.

    The physical joints are moved onto the reference pose at each environment's new phase
    to match; without that they would sit at the phase-0 pose returned by the reset event
    while ``gait_phase`` reports otherwise, until the policy's own tracking closes the gap.
    That transient is why this must run before, not instead of, the warmup window.

    The floating base is moved with the joints, for the same reason and with the same
    consequence if it is not: the reset event placed the root on the reference at phase 0,
    and leaving it there while the joints move to phase ``i / num_envs`` reintroduces
    exactly the pose-without-its-momentum mismatch that ``reset_to_reference_pose`` exists
    to remove -- worst of all for the ``--zero_actions`` reference playback, which has no
    policy to close the gap.
    """
    offsets = torch.arange(num_envs, device=env.device, dtype=torch.float32) / num_envs
    env.reference_gait.gait_phase[:] = offsets
    if env.cfg.events.reset_to_reference is not None:
        env.event_manager.get_term_cfg("reset_to_reference").params["randomize_phase"] = False

    robot = env.scene["robot"]
    q_ref, v_ref = env.reference_gait.sample()
    limits = robot.data.soft_joint_pos_limits.torch
    q_ref = torch.clamp(q_ref, limits[..., 0], limits[..., 1])
    robot.write_joint_state_to_sim(q_ref, v_ref)

    root_state = env.reference_gait.sample_root_state()
    if root_state is not None:
        height, quaternion, linear_velocity, angular_velocity = root_state
        pose = robot.data.root_pose_w.torch.clone()
        pose[:, 2] = env.scene.env_origins[:, 2] + height
        pose[:, 3:7] = quaternion
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(torch.cat([linear_velocity, angular_velocity], dim=-1))


def main() -> int:
    num_envs = max(2, args_cli.num_envs - args_cli.num_envs % 2)
    if num_envs != args_cli.num_envs:
        print(f"[evaluate] rounding {args_cli.num_envs} environments down to {num_envs} for a balanced paretic split")

    env_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg, selected=tuple(args_cli.presets))
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args_cli.seed
    # Long enough that the recording window is one uninterrupted episode: a timeout
    # mid-rollout would put a reset transient in the middle of the averaged curves.
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, (args_cli.num_steps + args_cli.warmup_steps + 10) * 0.02)

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    layout = env.joint_layout
    dt = float(env.step_dt)

    paretic_side = assign_balanced_paretic_sides(env, num_envs)
    is_right_paretic = (paretic_side > 0).cpu().numpy()

    policy = ActorCritic(
        obs_dim=int(env.observation_space["policy"].shape[-1]),
        action_dim=int(env.action_space.shape[-1]),
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
    ).to(env.device)
    if not args_cli.zero_actions and not args_cli.checkpoint:
        raise SystemExit("evaluate.py needs either --checkpoint or --zero_actions")

    if args_cli.zero_actions:
        # The action term is a residual on the reference pose, so zero actions command the
        # reference exactly. Rolling that out answers the question every reward term should
        # be checked against: what does the reference itself score here? A target the
        # reference cannot reach is one no policy tracking it can reach either.
        policy.eval()
        iteration = "reference"
        if args_cli.output_dir is None:
            raise SystemExit("--zero_actions needs an explicit --output_dir")
    else:
        checkpoint = torch.load(args_cli.checkpoint, map_location=env.device, weights_only=False)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.eval()
        iteration = checkpoint.get("iteration", "?")

    if args_cli.output_dir is None:
        output_dir = Path(args_cli.checkpoint).resolve().parent / "evaluation"
    else:
        output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args_cli.label or ("reference" if args_cli.zero_actions else Path(args_cli.checkpoint).stem)

    print("=" * 78)
    print(f"  Evaluation rollout -- {label} (checkpoint iteration {iteration})")
    print("=" * 78)
    print(f"  task {args_cli.task} | {num_envs} envs | {args_cli.num_steps} steps @ {dt * 1000:.0f} ms")
    print(f"  output -> {output_dir}", flush=True)

    foot_asset_cfg = SceneEntityCfg("robot", body_names=list(FOOT_BODY_NAMES), preserve_order=True)
    foot_asset_cfg.resolve(env.scene)
    foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_NAMES), preserve_order=True)
    foot_sensor_cfg.resolve(env.scene)

    obs, _ = env.reset()
    assign_phase_offsets(env, num_envs)
    obs = obs["policy"]

    recorder = RolloutRecorder()
    # An environment that terminates once is excluded from every later step: post-fall
    # kinematics are not gait, and letting them into the averages is what makes a fallen
    # policy look like it has an unusual gait pattern rather than no gait at all.
    ever_terminated = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    total_steps = args_cli.warmup_steps + args_cli.num_steps
    for step in range(total_steps):
        with torch.no_grad():
            if args_cli.zero_actions:
                actions = torch.zeros_like(env.action_manager.action)
            else:
                actions = policy.act(obs)[0] if args_cli.stochastic else policy.actor(obs)

        q_ref, v_ref = env.reference_gait.sample()
        gait_phase = env.reference_gait.gait_phase.clone()
        # The reference's own contact schedule at this phase, in (paretic, sound) order.
        # Recorded so schedule-gated metrics compare the policy against the reference over the
        # *same* window: the reference schedules 39.4% of the cycle as paretic swing while the
        # policy achieves 14-19%, so measured-swing and scheduled-swing averages are not
        # comparable quantities.
        ref_contact = env.reference_gait.sample_contact()
        if ref_contact is None:
            ref_contact = torch.zeros((env.num_envs, 2), device=env.device)

        obs, reward, terminated, truncated, _ = env.step(actions)
        obs = obs["policy"]
        ever_terminated |= terminated

        if step < args_cli.warmup_steps:
            continue

        xcom, mos, in_contact = compute_xcom_and_mos(env, foot_asset_cfg, foot_sensor_cfg)
        foot_force_w = env.scene["contact_forces"].data.net_forces_w.torch[:, foot_sensor_cfg.body_ids]
        foot_force = torch.norm(foot_force_w, dim=-1)
        # Fore-aft ground reaction force per foot, in the robot's own heading frame.
        #
        # The magnitude above discards direction, and direction is what paretic propulsion
        # needs: Bowden et al. (Stroke 37(3):872-876, 2006) define it as the paretic share of
        # the *anteriorly directed* A-P GRF impulse, 50% being symmetric, with patient values
        # of 16/36/49% for high/moderate/low severity. Two independent literature searches
        # named it the best-validated marker of post-stroke walking performance and the one a
        # simulation project is most likely to overlook -- and this project did overlook it
        # for nineteen runs, because the only force it stored was a norm.
        #
        # Rotated by the root yaw rather than taken as world x: episodes start at a random
        # heading, so world-frame fore-aft is meaningless across environments.
        heading = yaw_quat(robot.data.root_link_quat_w.torch)
        foot_force_ap = quat_apply_inverse(
            heading.unsqueeze(1).expand(-1, foot_force_w.shape[1], -1), foot_force_w
        )[..., 0]
        recorder.add(
            joint_pos=robot.data.joint_pos.torch,
            joint_vel=robot.data.joint_vel.torch,
            joint_ref_pos=q_ref,
            joint_ref_vel=v_ref,
            applied_torque=robot.data.applied_torque.torch,
            spastic_torque=env.applied_spastic_torque,
            root_pos=robot.data.root_link_pos_w.torch - env.scene.env_origins,
            root_quat=robot.data.root_link_quat_w.torch,
            root_lin_vel_b=robot.data.root_lin_vel_b.torch,
            root_ang_vel_b=robot.data.root_ang_vel_b.torch,
            foot_pos=robot.data.body_link_pos_w.torch[:, foot_asset_cfg.body_ids] - env.scene.env_origins[:, None, :],
            foot_force=foot_force,
            # Signed fore-aft GRF in the heading frame; positive is anterior (propulsive).
            foot_force_ap=foot_force_ap,
            foot_contact=(foot_force > 1.0),
            xcom=xcom,
            mos=mos,
            gait_phase=gait_phase,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            alive=~ever_terminated,
            ref_contact=ref_contact,
        )

        if (step + 1) % 200 == 0:
            print(f"[evaluate] step {step + 1 - args_cli.warmup_steps}/{args_cli.num_steps}", flush=True)

    data = recorder.stack()
    warn_if_undersampled(data, env.reference_gait.stride_duration_s, dt)
    # Read anything that lives behind a simulation handle before closing: the articulation
    # data views are weak references into the physics backend and raise once the app is
    # torn down. The layout tensors are already plain torch, so they survive.
    total_mass = float(robot.data.body_mass.torch[0].sum())
    env.close()

    try:
        metrics = summarize(data, layout, is_right_paretic, total_mass, dt, label, iteration)
    except NoSurvivingEnvironments as error:
        # A policy that falls in every environment is a legitimate result -- an early
        # checkpoint, or an ablation that does not learn to walk. Report it as a failed
        # condition rather than a traceback, so a batch of conditions keeps going.
        print(f"[evaluate] cannot summarize {label}: {error}", flush=True)
        np.savez_compressed(output_dir / "rollout.npz", **data, is_right_paretic=is_right_paretic, label=label)
        print(f"[evaluate] raw rollout kept at {output_dir / 'rollout.npz'} for inspection")
        return 1

    write_outputs(output_dir, data, metrics, layout, is_right_paretic, label)
    report(metrics)
    return 0


def warn_if_undersampled(data, stride_duration_s: float, dt: float) -> None:
    """Warn when the recording is too short to support the cycle-resolved metrics.

    The play config phase-locks every environment (``randomize_phase=False``), so cycle
    coverage comes from elapsed time alone -- 64 environments at the same phase fill the
    same bin. A rollout of one or two strides leaves most of the 101 cycle bins empty and
    yields too few complete stance periods for the timing metrics, which is how a short
    run produces a nonsensical toe-off percentage.
    """
    steps = data["gait_phase"].shape[0]
    strides = steps * dt / max(stride_duration_s, 1e-6)
    filled = len(np.unique(np.clip((data["gait_phase"] * (NUM_CYCLE_BINS - 1)).round().astype(int), 0, 100)))

    if strides < 8.0:
        print(
            f"[evaluate] WARNING: recorded only {strides:.1f} gait cycles; cycle-resolved curves and"
            f" the timing metrics want more. Use --num_steps >= {int(np.ceil(10 * stride_duration_s / dt))}"
            " for ~10 strides.",
            flush=True,
        )
    elif filled < NUM_CYCLE_BINS:
        print(
            f"[evaluate] WARNING: only {filled}/{NUM_CYCLE_BINS} cycle bins were sampled despite"
            f" {strides:.1f} recorded gait cycles. With --num_envs={data['gait_phase'].shape[1]}, phase"
            " offsets of 1/num_envs apart leave gaps; more --num_envs gives denser cycle coverage.",
            flush=True,
        )


def _ambulation_class(speed_ms: float) -> str:
    """Perry et al. (Stroke, 1995) walking-handicap class, validated by Bowden et al. (2008).

    Reported alongside every result because it is the context that decides whether a clinical
    normative range may be quoted at all. This project's own reference stride walks at
    0.244 m/s -- a household ambulator -- and its policies at 0.081 to 0.187, so every
    comparison against Patterson's 0.3-0.8+ m/s cohort is an extrapolation.
    """
    if not np.isfinite(speed_ms):
        return "unknown"
    if speed_ms < 0.40:
        return "household"
    if speed_ms <= 0.80:
        return "limited_community"
    return "community"


HIP_ROLL_LIMIT_DEG = 24.6
"""The H1's hip-roll joint limit (0.43 rad). Recorded values pile up against it exactly."""


def summarize(data, layout, is_right_paretic, total_mass, dt, label, iteration) -> dict[str, object]:
    """Reduce a recorded rollout to the scalar metrics the paper reports."""
    mirror_index = layout.mirror_index.cpu().numpy()
    mirror_sign = layout.mirror_sign.cpu().numpy()
    left_ids = layout.left_leg_ids.cpu().numpy()
    right_ids = layout.right_leg_ids.cpu().numpy()

    # After standardization the paretic limb is always in the left slots.
    q = standardize_to_paretic_frame(data["joint_pos"], is_right_paretic, mirror_index, mirror_sign)
    q_ref = standardize_to_paretic_frame(data["joint_ref_pos"], is_right_paretic, mirror_index, mirror_sign)

    valid = data["alive"].astype(bool)
    valid_steps = int(valid.sum())
    if valid_steps == 0:
        raise NoSurvivingEnvironments("every environment fell during the rollout; there is no gait to measure")

    rad2deg = 180.0 / np.pi
    error = (q - q_ref)[valid]
    joint_rmse_deg = np.sqrt(np.mean(np.square(error), axis=0)) * rad2deg

    # Gait Profile Score: the RMS of the lower-limb gait variable scores (Baker et al.).
    leg_ids = np.concatenate([left_ids, right_ids])
    gps_deg = float(np.sqrt(np.mean(np.square(joint_rmse_deg[leg_ids]))))

    # Cost of transport, from the robot's own mass and its actual displacement. The
    # actuators are implicit, so applied_torque is Isaac Lab's PD estimate rather than a
    # solver readback; the spastic feed-forward is added explicitly because it is real
    # work done against the paretic limb.
    torque = data["applied_torque"] + data["spastic_torque"]
    mechanical_power = np.abs(torque * data["joint_vel"]).sum(axis=-1)
    displacement = np.linalg.norm(data["root_pos"][-1, :, :2] - data["root_pos"][0, :, :2], axis=-1)
    alive_envs = valid[-1]
    work = (mechanical_power * valid).sum(axis=0) * dt
    with np.errstate(divide="ignore", invalid="ignore"):
        cot = work / (total_mass * GRAVITY * np.maximum(displacement, 1e-3))
    cost_of_transport = float(np.mean(cot[alive_envs])) if alive_envs.any() else float("nan")

    # Contact-derived timing, with the feet reordered so index 0 is the paretic one.
    contact = data["foot_contact"].astype(bool)
    paretic_first = np.where(is_right_paretic[None, :, None], contact[:, :, ::-1], contact)
    timing = contact_gait_metrics(paretic_first, valid, dt, gait_phase=data["gait_phase"])

    # Pelvic hiking: obliquity overall, and the swing-minus-stance signature that is the
    # hallmark itself. The reference carries +4.50 deg of it; a policy walking with a level
    # pelvis scores ~0 and one that drops the swing side scores negative.
    obliquity_deg, hiking_deg = pelvic_obliquity(
        data["root_quat"],
        is_right_paretic,
        valid,
        contact_schedule=data["ref_contact"][..., 0] < 0.5,
    )

    knee_paretic = layout.index_of("left_knee")
    knee_sound = layout.index_of("right_knee")
    ankle_paretic = layout.index_of("left_ankle")

    # Range of motion per environment, then a symmetry index across limbs. Taking the ROM
    # of the environment-averaged trace instead would understate it: environments at
    # different gait phases average toward a flatter curve.
    def rom(joint_index: int) -> np.ndarray:
        trace = np.where(valid, q[:, :, joint_index], np.nan)
        return (np.nanmax(trace, axis=0) - np.nanmin(trace, axis=0)) * rad2deg

    knee_rom_paretic = rom(knee_paretic)
    knee_rom_sound = rom(knee_sound)
    knee_si = 200.0 * np.abs(knee_rom_paretic - knee_rom_sound) / (knee_rom_paretic + knee_rom_sound + 1e-6)

    mos = data["mos"][valid]
    # With no foot loaded there is no base of support and Hof's margin is undefined; the
    # value carried for those samples is a limiting-case convention (see
    # compute_xcom_and_mos). Report the supported samples separately so a mean MoS is not
    # quietly an average over frames where the robot was in the air.
    grounded = data["foot_contact"].any(axis=-1)[valid]
    mos_grounded = mos[grounded]
    speed = data["root_lin_vel_b"][:, :, 0][valid]
    spastic = np.abs(data["spastic_torque"])
    survival = 100.0 * float(valid[-1].mean())

    # Share of total foot load carried by the paretic limb while both feet are down. This is
    # what ``paretic_load_aversion`` charges for, and reporting it separately is what makes
    # that term falsifiable: a policy can satisfy an aversion to paretic load by unloading
    # the foot while leaving it on the ground, which would move this number and leave
    # ``stance_fraction_asymmetry_pct`` where it was. Read together, the two separate a real
    # change in weight-transfer strategy from the term being gamed.
    #
    # 0.5 is symmetric loading. The reference's *time* asymmetry is -15.87%, so a policy
    # reproducing the strategy should sit below 0.5 here as well as shortening paretic stance.
    # Fraction of the recorded gait in which a hip roll sits against its mechanical stop.
    #
    # With the stance foot flat and the H1's ankle rigid in roll, the stance leg has exactly
    # one frontal-plane degree of freedom -- the hip roll -- and every configuration that
    # actually walks pins it at its +-24.6 deg limit for 62-75% of the cycle. The AMP arm is
    # the only exception and it avoids the stop by moving at a third of the reference speed.
    #
    # This went unreported for eighteen runs while `dof_pos_limits` quietly charged the policy
    # for it at weight -1.0 and the policy paid. A soft penalty on a hard constraint hides
    # exactly this: the trade "take the penalty, there is nowhere else to go". Mean pelvic
    # obliquity tracks the saturation at r = +0.73 within the no-AMP arm, which is why the
    # reproduced-pelvic-hiking claim now carries a qualification.
    hip_roll_columns = [
        index
        for index, name in enumerate(layout.sim_names)
        if name.endswith("hip_roll")
    ]
    if hip_roll_columns:
        hip_roll_deg = np.degrees(np.abs(data["joint_pos"][:, :, hip_roll_columns]))
        at_stop = (hip_roll_deg > HIP_ROLL_LIMIT_DEG - 1.0).any(axis=-1)
        hip_roll_saturation = float(np.mean(at_stop[valid])) if valid.any() else float("nan")
    else:
        hip_roll_saturation = float("nan")

    # Paretic propulsion, Bowden et al. (Stroke, 2006): the paretic share of the anteriorly
    # directed fore-aft GRF impulse. 50% is symmetric; their cohort scored 16/36/49% for
    # high/moderate/low hemiparetic severity. Only the positive (propulsive) part of the A-P
    # force counts -- the braking phase is a separate quantity and folding it in would cancel
    # most of the signal.
    #
    # Returns NaN for eval archives written before foot_force_ap was recorded, rather than
    # silently computing something else from the force magnitude.
    if "foot_force_ap" in data:
        ap_paretic_first = np.where(
            is_right_paretic[None, :, None], data["foot_force_ap"][:, :, ::-1], data["foot_force_ap"]
        )
        propulsive = np.clip(ap_paretic_first, 0.0, None) * valid[..., None]
        impulse = propulsive.sum(axis=0) * dt          # (num_envs, 2)
        total = impulse.sum(axis=-1)
        usable = total > 1e-6
        paretic_propulsion_pct = (
            float(100.0 * np.mean(impulse[usable, 0] / total[usable])) if usable.any() else float("nan")
        )
    else:
        paretic_propulsion_pct = float("nan")

    force_paretic_first = np.where(
        is_right_paretic[None, :, None], data["foot_force"][:, :, ::-1], data["foot_force"]
    )
    both_down = (force_paretic_first > 1.0).all(axis=-1) & valid
    if both_down.any():
        loads = force_paretic_first[both_down]
        paretic_load_share = float(np.mean(loads[:, 0] / np.maximum(loads.sum(axis=-1), 1e-6)))
    else:
        paretic_load_share = float("nan")

    return {
        "label": label,
        "checkpoint_iteration": iteration,
        "pelvic_obliquity_deg": obliquity_deg,
        # Paretic share of foot load during double support; 0.5 is symmetric.
        "paretic_load_share": paretic_load_share,
        # Fraction of the gait with a hip roll against its mechanical stop; see above.
        "hip_roll_saturation": hip_roll_saturation,
        # Bowden 2006: paretic share of the anterior A-P GRF impulse. 50% symmetric.
        "paretic_propulsion_pct": paretic_propulsion_pct,
        # Perry (1995) / Bowden (2008) walking-handicap class for the achieved speed. Every
        # normative comparison in the clinical literature is drawn from cohorts at
        # 0.3-0.8+ m/s, so a policy below 0.40 is outside the range the norms were measured in.
        "ambulation_class": _ambulation_class(float(np.mean(speed))),
        "pelvic_hiking_signature_deg": hiking_deg,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "num_envs": int(valid.shape[1]),
        "num_steps": int(valid.shape[0]),
        "valid_step_fraction": valid_steps / valid.size,
        # -- imitation fidelity
        "gait_profile_score_deg": gps_deg,
        "overall_joint_rmse_deg": float(np.mean(joint_rmse_deg)),
        "paretic_knee_rmse_deg": float(joint_rmse_deg[knee_paretic]),
        "sound_knee_rmse_deg": float(joint_rmse_deg[knee_sound]),
        "paretic_ankle_rmse_deg": float(joint_rmse_deg[ankle_paretic]),
        # -- clinical asymmetry
        "paretic_knee_rom_deg": float(np.nanmean(knee_rom_paretic)),
        "sound_knee_rom_deg": float(np.nanmean(knee_rom_sound)),
        "knee_symmetry_index_pct": float(np.nanmean(knee_si)),
        **timing,
        # -- locomotion
        "mean_forward_speed_ms": float(np.mean(speed)),
        "cost_of_transport": cost_of_transport,
        "total_mass_kg": total_mass,
        # -- balance
        "mean_mos_m": float(np.mean(mos_grounded)) if mos_grounded.size else float("nan"),
        "min_mos_m": float(np.min(mos_grounded)) if mos_grounded.size else float("nan"),
        "mos_positive_pct": float(100.0 * np.mean(mos_grounded > 0.0)) if mos_grounded.size else float("nan"),
        # Share of samples excluded from the three figures above, because no foot was
        # loaded. A large value makes them unrepresentative rather than merely noisy.
        "airborne_pct": float(100.0 * (1.0 - np.mean(grounded))),
        "mean_mos_all_samples_m": float(np.mean(mos)),
        # -- pathology
        "peak_spastic_torque_nm": float(spastic.max()),
        "mean_spastic_torque_nm": float(spastic[valid].mean()),
        "spastic_work_j_per_env": float(
            (np.abs(data["spastic_torque"] * data["joint_vel"]).sum(axis=-1) * valid).sum() * dt / valid.shape[1]
        ),
        "survival_pct": survival,
        "per_joint_rmse_deg": {name: float(joint_rmse_deg[index]) for index, name in enumerate(layout.sim_names)},
    }


def write_outputs(output_dir: Path, data, metrics, layout, is_right_paretic, label) -> None:
    """Write the rollout archive, the cycle-normalized curves, and the metrics tables."""
    mirror_index = layout.mirror_index.cpu().numpy()
    mirror_sign = layout.mirror_sign.cpu().numpy()
    valid = data["alive"].astype(bool)

    standardized = {
        f"{name}_std": standardize_to_paretic_frame(data[name], is_right_paretic, mirror_index, mirror_sign)
        for name in ("joint_pos", "joint_vel", "joint_ref_pos", "joint_ref_vel", "spastic_torque", "applied_torque")
    }

    np.savez_compressed(
        output_dir / "rollout.npz",
        **data,
        **standardized,
        is_right_paretic=is_right_paretic,
        sim_joint_names=np.array(layout.sim_names),
        clinical_joint_names=np.array(CLINICAL_JOINT_ORDER),
        label=label,
    )

    phase = data["gait_phase"]
    cycle = {}
    for name in ("joint_pos_std", "joint_ref_pos_std", "joint_vel_std", "spastic_torque_std"):
        mean, std = cycle_normalize(standardized[name], phase, valid)
        cycle[f"{name}_mean"] = mean
        cycle[f"{name}_std"] = std
    mos_mean, mos_std = cycle_normalize(data["mos"][..., None], phase, valid)
    contact_mean, _ = cycle_normalize(
        np.where(is_right_paretic[None, :, None], data["foot_contact"][:, :, ::-1], data["foot_contact"]).astype(float),
        phase,
        valid,
    )
    np.savez_compressed(
        output_dir / "gait_cycle.npz",
        cycle_pct=np.linspace(0.0, 100.0, NUM_CYCLE_BINS),
        mos_mean=mos_mean[:, 0],
        mos_std=mos_std[:, 0],
        # Contact probability per cycle bin; where it crosses 0.5 is toe-off, which is
        # what the swing shading in the kinematics figures keys off.
        paretic_contact_prob=contact_mean[:, 0],
        sound_contact_prob=contact_mean[:, 1],
        sim_joint_names=np.array(layout.sim_names),
        label=label,
        **cycle,
    )

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    flat = {key: value for key, value in metrics.items() if not isinstance(value, dict)}
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(flat.items())
        writer.writerow([])
        writer.writerow(["joint", "rmse_deg"])
        writer.writerows(metrics["per_joint_rmse_deg"].items())

    print(f"[evaluate] wrote rollout.npz, gait_cycle.npz, metrics.json, metrics.csv -> {output_dir}")


def report(metrics: dict) -> None:
    """Print the headline numbers."""
    rows = [
        ("Gait Profile Score", "gait_profile_score_deg", "deg"),
        ("Overall joint RMSE", "overall_joint_rmse_deg", "deg"),
        ("Paretic knee ROM", "paretic_knee_rom_deg", "deg"),
        ("Sound knee ROM", "sound_knee_rom_deg", "deg"),
        ("Knee symmetry index", "knee_symmetry_index_pct", "%"),
        ("Temporal asymmetry (signed)", "temporal_asymmetry_pct", "%"),
        ("Double support fraction", "double_support_fraction", ""),
        ("Stance periods per cycle (paretic)", "paretic_stance_periods_per_cycle", ""),
        ("Paretic stance fraction", "paretic_stance_fraction", ""),
        ("Sound stance fraction", "sound_stance_fraction", ""),
        ("Forward speed", "mean_forward_speed_ms", "m/s"),
        ("Cost of transport", "cost_of_transport", ""),
        ("Mean lateral MoS (grounded)", "mean_mos_m", "m"),
        ("Time with MoS > 0 (grounded)", "mos_positive_pct", "%"),
        ("Time airborne (MoS undefined)", "airborne_pct", "%"),
        ("Peak spastic torque", "peak_spastic_torque_nm", "Nm"),
        ("Survival", "survival_pct", "%"),
    ]
    print("\n" + "=" * 78)
    print(f"  {metrics['label']} -- {metrics['num_envs']} envs, {metrics['num_steps']} steps")
    print("=" * 78)
    for title, key, unit in rows:
        print(f"  {title:<26}: {metrics.get(key, float('nan')):10.4f} {unit}")
    print("=" * 78)


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
