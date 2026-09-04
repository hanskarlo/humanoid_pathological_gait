#!/usr/bin/env python3
"""
actor_critic.py

PPO Actor-Critic Policy Network and Vectorized Generalized Advantage Estimation (GAE) Rollout Buffer.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

DEFAULT_STATE_DIM = 107
DEFAULT_ACTION_DIM = 19


class ActorCritic(nn.Module):
    """
    Actor-Critic Network for continuous humanoid control.
      - Actor: Maps observation -> Gaussian action distribution mean mu(s).
      - Critic: Maps observation -> State value scalar V(s).
    """

    def __init__(
        self,
        obs_dim: int = DEFAULT_STATE_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        actor_hidden_dims: tuple[int, ...] = (512, 256, 128),
        critic_hidden_dims: tuple[int, ...] = (512, 256, 128),
        init_noise_std: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 1. Actor Network
        actor_layers = []
        in_dim = obs_dim
        for h_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(in_dim, h_dim))
            actor_layers.append(nn.ELU())
            in_dim = h_dim
        actor_layers.append(nn.Linear(in_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Action Log-Standard Deviation (Learnable Parameter)
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(init_noise_std))

        # 2. Critic Network
        critic_layers = []
        in_dim = obs_dim
        for h_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(in_dim, h_dim))
            critic_layers.append(nn.ELU())
            in_dim = h_dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor):
        raise NotImplementedError("Use act() or evaluate() instead.")

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples an action from the policy distribution.
        Returns: (actions, action_log_probs, state_values)
        """
        action_mean = self.actor(obs)
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)

        actions = dist.sample()
        action_log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.critic(obs).squeeze(-1)

        return actions, action_log_probs, values

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates actions under the current policy distribution.
        Returns: (action_log_probs, entropy, state_values)
        """
        action_mean = self.actor(obs)
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)

        action_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(obs).squeeze(-1)

        return action_log_probs, entropy, values


MiniBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
"""One PPO mini-batch: ``(obs, actions, old_log_probs, advantages, returns, values)``."""


class RolloutBuffer:
    """
    Stores on-policy rollout trajectories across parallel environments.
    Computes Generalized Advantage Estimation (GAE).
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs_buf = torch.zeros((num_steps, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.action_buf = torch.zeros((num_steps, num_envs, action_dim), dtype=torch.float32, device=device)
        self.log_prob_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.reward_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.value_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.dones_buf = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)

        self.advantages_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.returns_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

        self.step_idx = 0

    def insert(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
    ):
        self.obs_buf[self.step_idx] = obs
        self.action_buf[self.step_idx] = action
        self.log_prob_buf[self.step_idx] = log_prob
        self.reward_buf[self.step_idx] = reward
        self.value_buf[self.step_idx] = value
        self.dones_buf[self.step_idx] = done
        self.step_idx += 1

    def compute_returns_and_advantages(self, last_values: torch.Tensor, last_dones: torch.Tensor):
        last_gae_lam = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - last_dones.float()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones_buf[t + 1].float()
                next_values = self.value_buf[t + 1]

            delta = self.reward_buf[t] + self.gamma * next_values * next_non_terminal - self.value_buf[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages_buf[t] = last_gae_lam

        self.returns_buf = self.advantages_buf + self.value_buf
        self.step_idx = 0

    def get_mini_batch_generator(self, num_mini_batches: int = 4) -> Generator[MiniBatch, None, None]:
        total_samples = self.num_steps * self.num_envs
        batch_size = total_samples // num_mini_batches

        flat_obs = self.obs_buf.view(-1, self.obs_dim)
        flat_actions = self.action_buf.view(-1, self.action_dim)
        flat_log_probs = self.log_prob_buf.view(-1)
        flat_advantages = self.advantages_buf.view(-1)
        flat_returns = self.returns_buf.view(-1)
        flat_values = self.value_buf.view(-1)

        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        indices = torch.randperm(total_samples, device=self.device)
        for start_idx in range(0, total_samples, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            yield (
                flat_obs[batch_idx],
                flat_actions[batch_idx],
                flat_log_probs[batch_idx],
                flat_advantages[batch_idx],
                flat_returns[batch_idx],
                flat_values[batch_idx],
            )
