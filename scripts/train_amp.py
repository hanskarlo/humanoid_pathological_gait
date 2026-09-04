# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO + Asymmetric Adversarial Motion Prior training against the PhysX-backed H1 task.

This is the extension's real training route. The ``rsl_rl`` bundled with this Isaac Lab
release has no AMP support, so the policy is trained by the PPO+AMP loop in
``humanoid_pathological_gait.algorithms`` instead. Rewards come from the environment's own
reward manager; this loop adds only the adversarial style term.

The environment reward covers kinematic tracking, margin of stability and the whole-body
regularizers. This loop adds only the AMP style reward on top, weighted by ``--amp_weight``:

    r_total = r_env + w_amp * r_amp

The PPO and AMP components live in ``humanoid_pathological_gait.algorithms`` and are
env-agnostic, needing only ``(obs, reward, done)`` tensors and an AMP feature extractor.

.. note::
    The simulation app is launched before any task import; see ``zero_agent.py`` for why.
"""

import argparse

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-H1-Pathological-Gait-v0")
parser.add_argument("--num_envs", type=int, default=1024, help="Parallel simulation environments.")
parser.add_argument("--max_iterations", type=int, default=3500, help="Total training iterations.")
parser.add_argument("--num_steps_per_env", type=int, default=24, help="Rollout length per iteration.")
parser.add_argument("--save_interval", type=int, default=100, help="Checkpoint frequency, in iterations.")
parser.add_argument("--log_interval", type=int, default=10, help="Console logging frequency, in iterations.")
parser.add_argument("--amp_weight", type=float, default=5.0, help="Weight on the AMP style reward.")
parser.add_argument("--lr_policy", type=float, default=3e-4, help="Policy/value learning rate.")
parser.add_argument("--lr_disc", type=float, default=1e-4, help="Discriminator learning rate.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment and the learner.")
parser.add_argument("--log_dir", type=str, default="logs/ppo_amp", help="Root directory for run logs.")
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
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from isaaclab_tasks.utils import load_cfg_from_registry, resolve_presets

importlib.import_module("humanoid_pathological_gait.tasks")

from humanoid_pathological_gait.algorithms.amp import (  # noqa: E402
    AMP_TRANSITION_DIM,
    AMPAgentReplayBuffer,
    AMPDiscriminator,
    AMPExpertMotionBuffer,
    AMPLossManager,
    extract_amp_features,
)
from humanoid_pathological_gait.algorithms.ppo import ActorCritic, RolloutBuffer  # noqa: E402
from humanoid_pathological_gait.tasks.humanoid_pathological_gait.assets import expert_dataset_path  # noqa: E402


class PPOAMPTrainer:
    """PPO policy optimization with an adversarial motion-prior style reward."""

    def __init__(self, env, log_dir: Path, device: torch.device):
        self.env = env
        self.device = device
        self.log_dir = log_dir
        self.num_envs = env.num_envs
        self.num_steps = args_cli.num_steps_per_env
        self.num_epochs = 5
        self.num_mini_batches = 4
        self.clip_param = 0.2
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.005
        self.amp_weight = args_cli.amp_weight

        obs_dim = int(np.prod(env.observation_space["policy"].shape[1:]))
        action_dim = int(np.prod(env.action_space.shape[1:]))
        print(f"[train_amp] observation dim {obs_dim}, action dim {action_dim}")

        self.policy = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
        ).to(device)
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=args_cli.lr_policy)

        self.rollout_buffer = RolloutBuffer(
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )

        self.discriminator = AMPDiscriminator(input_dim=AMP_TRANSITION_DIM, hidden_dims=(512, 256)).to(device)
        self.disc_loss_mgr = AMPLossManager(
            self.discriminator, learning_rate=args_cli.lr_disc, gradient_penalty_weight=5.0
        )

        # The staged post-stroke corpus is the expert motion prior. Passing the path
        # explicitly keeps the buffer off its relative-path fallbacks.
        self.expert_buffer = AMPExpertMotionBuffer(dataset_path=str(expert_dataset_path()), device=device)
        self.agent_buffer = AMPAgentReplayBuffer(capacity=50_000, device=device)

        obs, _ = self.env.reset()
        self.current_obs = obs["policy"]

    def _amp_features(self) -> torch.Tensor:
        """Current AMP kinematic feature vector for every environment."""
        return extract_amp_features(*self.env.get_amp_kinematic_tensors())

    def train_iteration(self) -> dict[str, float]:
        """Collect one rollout, update the policy, then update the discriminator."""
        self.policy.eval()
        env_rewards, amp_rewards = [], []

        current_features = self._amp_features()
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for _ in range(self.num_steps):
            with torch.no_grad():
                actions, log_probs, values = self.policy.act(self.current_obs)

            next_obs, reward, terminated, truncated, _ = self.env.step(actions)
            dones = terminated | truncated

            next_features = self._amp_features()
            transitions = torch.cat([current_features, next_features], dim=-1)
            self.agent_buffer.insert(transitions)
            amp_reward = self.discriminator.compute_amp_reward(transitions)

            self.rollout_buffer.insert(
                obs=self.current_obs,
                action=actions,
                log_prob=log_probs,
                reward=reward + self.amp_weight * amp_reward,
                value=values,
                done=dones,
            )

            self.current_obs = next_obs["policy"]
            current_features = next_features
            env_rewards.append(float(reward.mean()))
            amp_rewards.append(float(amp_reward.mean()))

        with torch.no_grad():
            _, _, last_values = self.policy.act(self.current_obs)
        self.rollout_buffer.compute_returns_and_advantages(last_values, dones)

        # -- PPO update
        self.policy.train()
        policy_losses, value_losses, entropies = [], [], []
        for _ in range(self.num_epochs):
            for batch in self.rollout_buffer.get_mini_batch_generator(self.num_mini_batches):
                batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns, _ = batch
                log_probs, entropy, values = self.policy.evaluate(batch_obs, batch_actions)

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surrogate = torch.min(
                    ratio * batch_advantages,
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages,
                )
                policy_loss = -surrogate.mean()
                value_loss = 0.5 * torch.mean(torch.square(values - batch_returns))
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer_policy.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer_policy.step()

                policy_losses.append(float(policy_loss))
                value_losses.append(float(value_loss))
                entropies.append(float(entropy.mean()))

        # -- discriminator update
        disc_metrics: dict[str, float] = {}
        if self.agent_buffer.size >= 64:
            batch_size = min(256, self.agent_buffer.size)
            for _ in range(2):
                disc_metrics = self.disc_loss_mgr.train_step(
                    self.expert_buffer.sample_transitions(batch_size),
                    self.agent_buffer.sample(batch_size),
                )

        return {
            "env_reward": float(np.mean(env_rewards)),
            "amp_reward": float(np.mean(amp_rewards)),
            "episode_length": float(self.env.episode_length_buf.float().mean()),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            **disc_metrics,
        }

    def save_checkpoint(self, iteration: int) -> Path:
        """Write policy and discriminator state to the run directory."""
        path = self.log_dir / f"model_{iteration}.pt"
        torch.save(
            {
                "iteration": iteration,
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_policy": self.optimizer_policy.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "optimizer_disc": self.disc_loss_mgr.optimizer.state_dict(),
            },
            path,
        )
        return path


def main() -> int:
    env_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg, selected=tuple(args_cli.presets))
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)

    if args_cli.run_dir:
        log_dir = Path(args_cli.run_dir).absolute()
    else:
        log_dir = Path(args_cli.log_dir).absolute() / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    trainer = PPOAMPTrainer(env, log_dir, torch.device(env.device))

    print("=" * 80)
    print("  19-DoF H1 post-stroke gait -- PPO + Asymmetric AMP (Isaac Sim / PhysX)")
    print("=" * 80)
    print(f"  task {args_cli.task} | envs {args_cli.num_envs} | device {env.device}")
    print(f"  amp weight {args_cli.amp_weight} | expert trajectories {len(trainer.expert_buffer.trajectories)}")
    print(f"  logs -> {log_dir}")

    start = time.time()
    for iteration in range(1, args_cli.max_iterations + 1):
        metrics = trainer.train_iteration()

        if iteration % args_cli.log_interval == 0 or iteration == 1 or iteration == args_cli.max_iterations:
            print(
                f"[iter {iteration:05d}/{args_cli.max_iterations} | {time.time() - start:7.1f}s]"
                f" env_r {metrics['env_reward']:8.3f}"
                f" | amp_r {metrics['amp_reward']:6.3f}"
                f" | ep_len {metrics['episode_length']:6.1f}"
                f" | pol {metrics['policy_loss']:8.4f}"
                f" | val {metrics['value_loss']:9.3f}"
                f" | disc {metrics.get('disc_total_loss', float('nan')):7.4f}"
            )

        if iteration % args_cli.save_interval == 0 or iteration == args_cli.max_iterations:
            print(f"[train_amp] checkpoint: {trainer.save_checkpoint(iteration)}")

    env.close()
    return 0


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
