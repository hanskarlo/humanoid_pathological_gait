# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play back a trained PPO+AMP checkpoint and report clinical gait metrics.

Runs the deterministic ``-Play-v0`` task by default: fixed start phase, no pushes, no
observation noise, spasticity and paretic weakness at full strength. That is the
configuration to evaluate in -- the randomization in the training task exists to make the
policy robust, not to describe the gait.

Actions are the policy mean rather than a sample, so repeated runs of the same checkpoint
are comparable.

.. note::
    The simulation app is launched before any task import; see ``zero_agent.py`` for why.
"""

import argparse

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a train_amp.py checkpoint (.pt).")
parser.add_argument("--task", type=str, default="Isaac-H1-Pathological-Gait-Play-v0")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=600, help="Control steps to roll out.")
parser.add_argument("--video", action="store_true", help="Record a video of the rollout.")
parser.add_argument("--video_length", type=int, default=400, help="Video length in control steps.")
parser.add_argument(
    "--presets", type=str, nargs="*", default=(), help="Preset variants to select, e.g. --presets newton_mjwarp."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs only once the simulation app is up."""

import importlib
import os

import gymnasium as gym
import torch

from isaaclab_tasks.utils import load_cfg_from_registry, resolve_presets

importlib.import_module("humanoid_pathological_gait.tasks")

from humanoid_pathological_gait.algorithms.ppo import ActorCritic  # noqa: E402


def main() -> int:
    env_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg, selected=tuple(args_cli.presets))
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_dir = os.path.join(os.path.dirname(os.path.abspath(args_cli.checkpoint)), "videos")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[play] recording video to {video_dir}")

    base_env = env.unwrapped
    obs_dim = int(base_env.observation_space["policy"].shape[-1])
    action_dim = int(base_env.action_space.shape[-1])

    policy = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
    ).to(base_env.device)
    checkpoint = torch.load(args_cli.checkpoint, map_location=base_env.device, weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    print(f"[play] loaded {args_cli.checkpoint} (iteration {checkpoint.get('iteration', '?')})")

    obs, _ = env.reset()
    obs = obs["policy"]

    # Accumulate the quantities a clinician would ask about.
    tracking_errors, forward_speeds, mos_values, spastic_torques = [], [], [], []
    episode_returns = torch.zeros(base_env.num_envs, device=base_env.device)
    # Count steps here rather than reading env.episode_length_buf: step() resets terminated
    # environments internally, so that buffer is already back to zero when step() returns.
    episode_steps = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
    completed_returns, completed_lengths = [], []

    from isaaclab.managers import SceneEntityCfg

    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.mdp.rewards import compute_xcom_and_mos

    foot_names = list(base_env.cfg.rewards.margin_of_stability.params["asset_cfg"].body_names)
    foot_asset_cfg = SceneEntityCfg("robot", body_names=foot_names, preserve_order=True)
    foot_asset_cfg.resolve(base_env.scene)
    foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=foot_names, preserve_order=True)
    foot_sensor_cfg.resolve(base_env.scene)

    robot = base_env.scene["robot"]
    for _ in range(args_cli.num_steps):
        with torch.no_grad():
            # Mean action, not a sample: evaluation should be deterministic.
            actions = policy.actor(obs)

        obs, reward, terminated, truncated, _ = env.step(actions)
        obs = obs["policy"]
        episode_returns += reward
        episode_steps += 1

        dones = terminated | truncated
        if dones.any():
            finished = dones.nonzero(as_tuple=False).squeeze(-1)
            completed_returns.extend(episode_returns[finished].tolist())
            completed_lengths.extend(episode_steps[finished].tolist())
            episode_returns[finished] = 0.0
            episode_steps[finished] = 0

        q_ref, _ = base_env.reference_gait.sample()
        weights = base_env.joint_layout.tracking_weights
        weighted_mse = torch.sum(torch.square(robot.data.joint_pos.torch - q_ref) * weights, dim=-1) / weights.sum()
        tracking_errors.append(torch.sqrt(weighted_mse).mean())
        forward_speeds.append(robot.data.root_lin_vel_b.torch[:, 0].mean())
        _, mos, _ = compute_xcom_and_mos(base_env, foot_asset_cfg, foot_sensor_cfg)
        mos_values.append(mos.mean())
        spastic_torques.append(base_env.applied_spastic_torque.abs().amax(dim=-1).mean())

    def mean(values: list[torch.Tensor]) -> float:
        return float(torch.stack(values).mean())

    print("\n" + "=" * 68)
    print(f"  Playback summary -- {args_cli.task}, {args_cli.num_envs} envs, {args_cli.num_steps} steps")
    print("=" * 68)
    print(f"  weighted RMS joint tracking error : {mean(tracking_errors):8.4f} rad")
    print(f"  forward speed                     : {mean(forward_speeds):8.4f} m/s")
    print(f"  mediolateral margin of stability  : {mean(mos_values):8.4f} m")
    print(f"  peak paretic reflex torque        : {mean(spastic_torques):8.3f} Nm")
    if completed_lengths:
        mean_length = sum(completed_lengths) / len(completed_lengths)
        mean_return = sum(completed_returns) / len(completed_returns)
        print(f"  episodes completed                : {len(completed_lengths)}")
        print(f"  mean episode length               : {mean_length:8.1f} steps")
        print(f"  mean episode return               : {mean_return:8.2f}")
    else:
        print("  episodes completed                :        0 (no episode ended in this window)")
    print("=" * 68)

    env.close()
    return 0


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
