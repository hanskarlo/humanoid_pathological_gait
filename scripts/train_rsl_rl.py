# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stock RSL-RL PPO training, used as an environment sanity check.

This is *not* the project's training route. rsl_rl in this Isaac Lab release has no
AMP support, and the pathological-gait policy is trained by the custom PPO+AMP loop in
``train_amp.py``. What this script buys is a way to prove the environment's managers,
rewards and terminations drive a standard learner without shape or NaN problems,
independently of the AMP machinery.

The ``isaaclab train`` console command cannot serve this purpose here: the installed
``isaaclab`` package ships no ``scripts/reinforcement_learning`` directory for its CLI
to dispatch to.

.. note::
    The simulation app is launched before any task import; see ``zero_agent.py`` for why.
"""

import argparse

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-H1-Pathological-Gait-v0")
parser.add_argument("--num_envs", type=int, default=None, help="Override the configured environment count.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override the configured iteration count.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment and the learner.")
parser.add_argument("--log_dir", type=str, default="logs/rsl_rl", help="Root directory for run logs.")
parser.add_argument(
    "--run_dir", type=str, default=None, help="Exact run directory; overrides the timestamped one under --log_dir."
)
parser.add_argument(
    "--presets", type=str, nargs="*", default=(), help="Preset variants to select, e.g. --presets newton_mjwarp."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs only once the simulation app is up."""

import importlib
import importlib.metadata
import os
from datetime import datetime

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import load_cfg_from_registry, resolve_presets

importlib.import_module("humanoid_pathological_gait.tasks")


def main() -> int:
    env_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg, selected=tuple(args_cli.presets))
    env_cfg.sim.device = args_cli.device
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    agent_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "rsl_rl_cfg_entry_point")
    # RslRl*Cfg still declares the pre-5.0 `stochastic`/`init_noise_std` fields as MISSING;
    # this drops them so the config matches the installed rsl-rl's model constructors.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed

    if args_cli.run_dir:
        run_dir = os.path.abspath(args_cli.run_dir)
    else:
        run_dir = os.path.join(
            os.path.abspath(args_cli.log_dir),
            agent_cfg.experiment_name,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
    os.makedirs(run_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=run_dir, device=agent_cfg.device)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"[train_rsl_rl] logs and checkpoints written to {run_dir}")
    env.close()
    return 0


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
