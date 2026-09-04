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
parser.add_argument(
    "--no_tensorboard", action="store_true", help="Skip the TensorBoard event file; metrics.csv is always written."
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Checkpoint to resume from, or a run directory to resume from its newest checkpoint. "
    "Training continues at the stored iteration and metrics.csv is appended to, not truncated.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs only once the simulation app is up."""

import csv
import importlib
import json
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


class MetricWriter:
    """Per-iteration metric sink: a CSV for the paper's figures, TensorBoard for watching.

    The CSV is the artifact that matters. Console output is sampled every
    ``--log_interval`` iterations and is not a reliable record -- it is line-buffered
    through a redirect, is easy to lose to a truncated pipe, and drops the per-reward-term
    breakdown entirely. Every iteration is written here instead, flushed as it goes, so a
    run that is killed at iteration 2900 still yields a complete training curve.

    The column set is **not** frozen from the first row. The reward manager only publishes
    its per-term episodic sums once environments have started resetting, so a term can
    first appear at iteration 2 or later; a header fixed at iteration 1 would silently
    drop exactly the per-term breakdown this writer exists to capture. Instead the file is
    rewritten whenever a new column shows up -- a few thousand rows is nothing to rewrite,
    and it happens only in the first handful of iterations.
    """

    def __init__(self, log_dir: Path, enable_tensorboard: bool = True, resume_after: int = 0):
        self.csv_path = log_dir / "metrics.csv"
        self._fieldnames: list[str] = []
        self._rows: list[dict[str, float]] = []

        # Resuming into the same directory must not discard the earlier iterations: they
        # are the first half of the training curve. Rows at or beyond the resume point are
        # dropped, since those iterations are about to be recomputed.
        if resume_after and self.csv_path.exists():
            with self.csv_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                self._fieldnames = list(reader.fieldnames or [])
                self._rows = [row for row in reader if int(float(row["iteration"])) <= resume_after]
            print(f"[train_amp] carried {len(self._rows)} earlier metric rows into the resumed run")

        self.tb_writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(log_dir=str(log_dir))
            except ImportError:
                print("[train_amp] tensorboard not installed; writing metrics.csv only")

    @property
    def last_wall_time_s(self) -> float:
        """Wall time of the newest carried-over row, or zero for a fresh run."""
        return float(self._rows[-1]["wall_time_s"]) if self._rows else 0.0

    def write(self, iteration: int, wall_time_s: float, metrics: dict[str, float]) -> None:
        """Record one iteration's metrics and flush the file to disk."""
        row: dict[str, float] = {"iteration": iteration, "wall_time_s": round(wall_time_s, 3)}
        row.update({key: float(value) for key, value in metrics.items()})
        self._rows.append(row)

        new_columns = [key for key in row if key not in self._fieldnames]
        if new_columns:
            self._fieldnames.extend(new_columns)
            self._rewrite()
        else:
            with self.csv_path.open("a", newline="") as handle:
                # restval: a metric absent this iteration (the discriminator sits out the
                # first few) must leave a blank cell, not shift every later column left.
                csv.DictWriter(handle, fieldnames=self._fieldnames, restval="").writerow(row)

        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, iteration)

    def _rewrite(self) -> None:
        """Rewrite the whole file under the current column set."""
        with self.csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames, restval="")
            writer.writeheader()
            writer.writerows(self._rows)

    def close(self) -> None:
        """Flush the final state and close the event file."""
        self._rewrite()
        if self.tb_writer is not None:
            self.tb_writer.close()


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
        # Per-reward-term episodic sums, accumulated across the rollout. The reward
        # manager refreshes these in extras["log"] only when environments reset, so a
        # single step's snapshot is stale for most of the batch; averaging over the
        # rollout gives the term breakdown a reward-ablation figure needs.
        term_totals: dict[str, float] = {}
        term_counts: dict[str, int] = {}
        num_terminated = 0
        num_truncated = 0

        current_features = self._amp_features()
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for _ in range(self.num_steps):
            with torch.no_grad():
                actions, log_probs, values = self.policy.act(self.current_obs)

            next_obs, reward, terminated, truncated, extras = self.env.step(actions)
            dones = terminated | truncated
            num_terminated += int(terminated.sum())
            num_truncated += int(truncated.sum())

            for key, value in extras.get("log", {}).items():
                scalar = float(value) if not torch.is_tensor(value) else float(value.mean())
                if np.isfinite(scalar):
                    term_totals[key] = term_totals.get(key, 0.0) + scalar
                    term_counts[key] = term_counts.get(key, 0) + 1

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

                # detach() before the scalar conversion: these are live graph nodes, and
                # float() on one warns on every mini-batch of every iteration.
                policy_losses.append(float(policy_loss.detach()))
                value_losses.append(float(value_loss.detach()))
                entropies.append(float(entropy.mean().detach()))

        # -- discriminator update
        disc_metrics: dict[str, float] = {}
        if self.agent_buffer.size >= 64:
            batch_size = min(256, self.agent_buffer.size)
            for _ in range(2):
                disc_metrics = self.disc_loss_mgr.train_step(
                    self.expert_buffer.sample_transitions(batch_size),
                    self.agent_buffer.sample(batch_size),
                )

        steps_this_iter = self.num_steps * self.num_envs
        return {
            "env_reward": float(np.mean(env_rewards)),
            "amp_reward": float(np.mean(amp_rewards)),
            "total_reward": float(np.mean(env_rewards) + self.amp_weight * np.mean(amp_rewards)),
            "episode_length": float(self.env.episode_length_buf.float().mean()),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            # Termination rate separates "fell over" from "ran out of clock", which the
            # episode-length mean alone conflates.
            "termination_rate": num_terminated / steps_this_iter,
            "timeout_rate": num_truncated / steps_this_iter,
            # Policy exploration noise: a collapsing action_std with a flat reward is the
            # signature of premature convergence, and is invisible in the reward curve.
            "action_std": float(torch.exp(self.policy.log_std).mean()),
            **disc_metrics,
            **{key: total / term_counts[key] for key, total in term_totals.items()},
        }

    def load_checkpoint(self, path: Path) -> int:
        """Restore policy, discriminator and both optimizers. Returns the stored iteration.

        The optimizer states matter as much as the weights: Adam's moment estimates take
        hundreds of iterations to rebuild, and a resume that restores weights alone shows
        up as a visible dip in the training curve at the resume point.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer_policy.load_state_dict(checkpoint["optimizer_policy"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.disc_loss_mgr.optimizer.load_state_dict(checkpoint["optimizer_disc"])

        # The observation buffer was captured before the load; re-reading it keeps the
        # first resumed rollout consistent with the restored policy.
        obs, _ = self.env.reset()
        self.current_obs = obs["policy"]
        return int(checkpoint.get("iteration", 0))

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


def resolve_resume_checkpoint(target: str) -> Path:
    """Accept either a checkpoint path or a run directory, returning a checkpoint.

    Pointing at a directory is the common case after a crash -- the caller knows which run
    died, not which iteration it reached -- so the newest ``model_*.pt`` is selected by
    iteration number rather than by mtime, which reorders under a filesystem copy.
    """
    path = Path(target)
    if path.is_file():
        return path
    if not path.is_dir():
        raise SystemExit(f"[train_amp] --resume target does not exist: {target}")

    checkpoints = sorted(path.glob("model_*.pt"), key=lambda item: int(item.stem.split("_")[-1]))
    if not checkpoints:
        raise SystemExit(f"[train_amp] no model_*.pt found in {target}")
    return checkpoints[-1]


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

    start_iteration = 0
    if args_cli.resume:
        resume_path = resolve_resume_checkpoint(args_cli.resume)
        start_iteration = trainer.load_checkpoint(resume_path)
        print(f"[train_amp] resumed {resume_path} at iteration {start_iteration}", flush=True)

    writer = MetricWriter(log_dir, enable_tensorboard=not args_cli.no_tensorboard, resume_after=start_iteration)

    # Freeze the run's provenance next to its metrics. A figure caption has to state the
    # seed and environment count, and recovering those from a shell's scrollback later is
    # how runs become unreportable.
    run_config = {
        **vars(args_cli),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "resumed_from_iteration": start_iteration,
        "device": str(env.device),
        "num_expert_trajectories": len(trainer.expert_buffer.trajectories),
        "reward_terms": {
            name: env.reward_manager.get_term_cfg(name).weight for name in env.reward_manager.active_terms
        },
        "episode_length_s": env.cfg.episode_length_s,
        "step_dt": env.step_dt,
    }
    (log_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    print("=" * 80)
    print("  19-DoF H1 post-stroke gait -- PPO + Asymmetric AMP (Isaac Sim / PhysX)")
    print("=" * 80)
    print(f"  task {args_cli.task} | envs {args_cli.num_envs} | device {env.device}")
    print(f"  amp weight {args_cli.amp_weight} | expert trajectories {len(trainer.expert_buffer.trajectories)}")
    print(f"  logs -> {log_dir}")
    print(f"  metrics -> {writer.csv_path}", flush=True)

    if start_iteration >= args_cli.max_iterations:
        print(f"[train_amp] checkpoint is already at iteration {start_iteration}; nothing to do")
        env.close()
        return 0

    # Continue the wall-clock axis from where the previous run left off, so a resumed
    # run's wall_time_s stays monotonic instead of restarting at zero mid-curve.
    elapsed_offset = float(writer.last_wall_time_s)

    start = time.time()
    for iteration in range(start_iteration + 1, args_cli.max_iterations + 1):
        metrics = trainer.train_iteration()
        writer.write(iteration, elapsed_offset + time.time() - start, metrics)

        if iteration % args_cli.log_interval == 0 or iteration == 1 or iteration == args_cli.max_iterations:
            print(
                f"[iter {iteration:05d}/{args_cli.max_iterations} | {time.time() - start:7.1f}s]"
                f" env_r {metrics['env_reward']:8.3f}"
                f" | amp_r {metrics['amp_reward']:6.3f}"
                f" | ep_len {metrics['episode_length']:6.1f}"
                f" | pol {metrics['policy_loss']:8.4f}"
                f" | val {metrics['value_loss']:9.3f}"
                f" | disc {metrics.get('disc_total_loss', float('nan')):7.4f}",
                # Unbuffered: the console log is usually redirected to a file, and a
                # 4 KB buffer leaves it minutes behind the run it is meant to report.
                flush=True,
            )

        if iteration % args_cli.save_interval == 0 or iteration == args_cli.max_iterations:
            print(f"[train_amp] checkpoint: {trainer.save_checkpoint(iteration)}", flush=True)

    writer.close()
    env.close()
    return 0


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
