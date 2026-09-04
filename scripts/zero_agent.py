# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a registered task with a zero-action agent and report obs/reward health.

The ``isaaclab`` console script shipped with Isaac Lab 3.0.0b2 only dispatches
``train``/``play``, so this project carries its own zero-agent entry point.

Note the import order below. On this Isaac Sim 6.0.1 build, importing
``isaaclab.envs.ManagerBasedRLEnv`` -- which loading any task config transitively
does -- before ``AppLauncher`` starts Kit leaves the USD/``pxr`` bindings in a state
Kit cannot start from ("Caught an unknown exception!" out of ``app.startup``). So the
simulation app is launched first and every task import happens after it, which is also
the long-standing Isaac Lab standalone-script convention.
"""

import argparse

# Warp captures ``enable_backward`` at module creation (import) time. Isaac Lab does not
# use Warp autodiff, and skipping adjoint codegen roughly halves cold kernel-cache builds.
import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-H1-Pathological-Gait-v0", help="Registered gym id to run.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=200, help="Number of control steps to run.")
parser.add_argument(
    "--presets", type=str, nargs="*", default=(), help="Preset variants to select, e.g. --presets newton_mjwarp."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs only once the simulation app is up."""

import importlib

import gymnasium as gym
import torch

from isaaclab_tasks.utils import load_cfg_from_registry, resolve_presets

importlib.import_module("humanoid_pathological_gait.tasks")


def _tensors(obs) -> list[torch.Tensor]:
    """Flatten whatever the env returned as observations into a list of tensors."""
    if isinstance(obs, torch.Tensor):
        return [obs]
    if isinstance(obs, dict):
        return [t for v in obs.values() for t in _tensors(v)]
    if isinstance(obs, (list, tuple)):
        return [t for v in obs for t in _tensors(v)]
    return []


def main() -> int:
    env_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg, selected=tuple(args_cli.presets))
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    num_bad = 0
    reward_sum = 0.0
    reward_min = float("inf")
    reward_max = float("-inf")
    for step in range(args_cli.num_steps):
        actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        obs, reward, terminated, truncated, _ = env.step(actions)

        bad = [t for t in _tensors(obs) if not torch.isfinite(t).all()]
        if bad or not torch.isfinite(reward).all():
            num_bad += 1
            if num_bad == 1:
                print(f"[zero_agent] NON-FINITE obs/reward first seen at step {step}")
        reward_sum += float(reward.mean())
        reward_min = min(reward_min, float(reward.min()))
        reward_max = max(reward_max, float(reward.max()))

    dims = {k: tuple(v.shape) for k, v in obs.items()} if isinstance(obs, dict) else tuple(obs.shape)
    print(f"[zero_agent] task={args_cli.task} envs={args_cli.num_envs} steps={args_cli.num_steps}")
    print(f"[zero_agent] obs shapes: {dims}")
    print(
        f"[zero_agent] reward per step: mean={reward_sum / max(args_cli.num_steps, 1):.4f}"
        f" min={reward_min:.4f} max={reward_max:.4f}"
    )
    print(f"[zero_agent] steps with non-finite obs/reward: {num_bad}")

    env.close()
    return 1 if num_bad else 0


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
