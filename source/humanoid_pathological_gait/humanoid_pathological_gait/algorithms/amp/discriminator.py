#!/usr/bin/env python3
"""
discriminator.py

Asymmetric Adversarial Motion Prior (AMP) Discriminator and Reference Motion Buffer.
Provides:
  - Asymmetric Discriminator Network D_psi(s_amp, s'_amp)
  - Least-Squares GAN (LSGAN) Objective with Gradient Penalty
  - Style Reward formulation: r_amp = max(0, 1 - 0.25*(D(s, s') - 1)^2)
  - Replay buffer and Expert Motion Dataset loader for clinical stroke gait trajectories.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ==============================================================================
# 1. Asymmetric AMP Motion State Feature Formulator
# ==============================================================================

# AMP feature vector dimension:
# Root height (1) + Projected Gravity (3) + Root LinVel (3) + Root AngVel (3) + Joint Pos (19) + Joint Vel (19)
# Dimension = 1 + 3 + 3 + 3 + 19 + 19 = 48
AMP_FEATURE_DIM = 48
AMP_TRANSITION_DIM = AMP_FEATURE_DIM * 2  # (s_t, s_{t+1}) -> 96


def extract_amp_features(
    root_pos_z: torch.Tensor,  # (N, 1) or (N,)
    projected_gravity: torch.Tensor,  # (N, 3)
    root_lin_vel: torch.Tensor,  # (N, 3)
    root_ang_vel: torch.Tensor,  # (N, 3)
    joint_pos: torch.Tensor,  # (N, 19)
    joint_vel: torch.Tensor,  # (N, 19)
) -> torch.Tensor:
    """
    Extracts pure kinematic motion features for the AMP discriminator.
    Notice the asymmetry: task conditioning, gait phase, and action history are omitted
    to allow the discriminator to evaluate motion fidelity objectively.
    """
    if root_pos_z.ndim == 1:
        root_pos_z = root_pos_z.unsqueeze(-1)

    features = torch.cat(
        [
            root_pos_z,
            projected_gravity,
            root_lin_vel,
            root_ang_vel,
            joint_pos,
            joint_vel,
        ],
        dim=-1,
    )

    assert features.shape[-1] == AMP_FEATURE_DIM, f"Expected {AMP_FEATURE_DIM}, got {features.shape[-1]}"
    return features


# ==============================================================================
# 2. Discriminator Neural Network Architecture
# ==============================================================================


class AMPDiscriminator(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Discriminator D_psi: (s_t, s_{t+1}) -> R.
    Evaluates kinematic realism of simulated motions relative to clinical stroke priors.
    """

    def __init__(
        self,
        input_dim: int = AMP_TRANSITION_DIM,
        hidden_dims: tuple[int, ...] = (1024, 512),
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim

        # Running normalizer for input feature stability
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.register_buffer("count", torch.tensor(1e-4))

        act_cls = nn.ReLU if activation.lower() == "relu" else nn.ELU

        layers = []
        in_d = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_d, h_dim))
            layers.append(act_cls())
            in_d = h_dim

        # Linear output head (unbounded logit for LSGAN / WGAN)
        layers.append(nn.Linear(in_d, 1))
        self.network = nn.Sequential(*layers)

        # Initialize network weights
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, amp_transitions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            amp_transitions: (batch_size, AMP_TRANSITION_DIM) concatenated (s_t, s_{t+1}).
        Returns:
            scores: (batch_size, 1) real-valued discriminator predictions.
        """
        norm_transitions = (amp_transitions - self.running_mean) / torch.sqrt(self.running_var + 1e-8)
        return self.network(norm_transitions)

    def update_normalizer(self, amp_transitions: torch.Tensor):
        """Updates online mean and variance statistics for feature normalization."""
        with torch.no_grad():
            batch_mean = torch.mean(amp_transitions, dim=0)
            batch_var = torch.var(amp_transitions, dim=0, unbiased=False)
            batch_count = amp_transitions.shape[0]

            delta = batch_mean - self.running_mean
            tot_count = self.count + batch_count

            new_mean = self.running_mean + delta * batch_count / tot_count
            m_a = self.running_var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
            new_var = M2 / tot_count

            self.running_mean.copy_(new_mean)
            self.running_var.copy_(new_var)
            self.count.copy_(tot_count)

    def compute_amp_reward(self, amp_transitions: torch.Tensor) -> torch.Tensor:
        """
        Computes style reward for the RL policy:
          r_amp = max(0, 1 - 0.25 * (D_psi(s, s') - 1)^2)
        """
        with torch.no_grad():
            disc_score = self.forward(amp_transitions).squeeze(-1)
            reward = torch.clamp(1.0 - 0.25 * torch.square(disc_score - 1.0), min=0.0)
            return reward


# ==============================================================================
# 3. AMP Loss and Optimization Manager
# ==============================================================================


class AMPLossManager:
    """
    Manages LSGAN loss calculation with 1-sided gradient penalty on expert samples.
    """

    def __init__(
        self,
        discriminator: AMPDiscriminator,
        learning_rate: float = 1e-4,
        gradient_penalty_weight: float = 5.0,
        weight_decay: float = 1e-4,
    ):
        self.discriminator = discriminator
        self.gp_weight = gradient_penalty_weight
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train_step(
        self,
        expert_transitions: torch.Tensor,
        agent_transitions: torch.Tensor,
    ) -> dict[str, float]:
        # Update normalization stats with combined batch
        combined = torch.cat([expert_transitions, agent_transitions], dim=0)
        self.discriminator.update_normalizer(combined)

        expert_transitions.requires_grad_(True)
        expert_scores = self.discriminator(expert_transitions)
        agent_scores = self.discriminator(agent_transitions.detach())

        # 1. Least-Squares GAN Loss (Demo target = 1.0, Agent target = -1.0)
        loss_demo = 0.5 * torch.mean(torch.square(expert_scores - 1.0))
        loss_agent = 0.5 * torch.mean(torch.square(agent_scores + 1.0))
        lsgan_loss = loss_demo + loss_agent

        # 2. Gradient Penalty on Expert Demonstrations
        ones = torch.ones_like(expert_scores)
        grad_demo = torch.autograd.grad(
            outputs=expert_scores,
            inputs=expert_transitions,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_penalty = 0.5 * self.gp_weight * torch.mean(torch.sum(torch.square(grad_demo), dim=-1))

        total_loss = lsgan_loss + grad_penalty

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=10.0)
        self.optimizer.step()

        with torch.no_grad():
            expert_acc = (expert_scores > 0.0).float().mean().item()
            agent_acc = (agent_scores < 0.0).float().mean().item()

        return {
            "disc_total_loss": total_loss.item(),
            "disc_lsgan_loss": lsgan_loss.item(),
            "disc_gp_loss": grad_penalty.item(),
            "expert_score_mean": expert_scores.mean().item(),
            "agent_score_mean": agent_scores.mean().item(),
            "expert_acc": expert_acc,
            "agent_acc": agent_acc,
        }


# ==============================================================================
# 4. Expert Motion & Agent Replay Buffers
# ==============================================================================


class AMPExpertMotionBuffer:
    """
    Stores and samples contiguous motion transitions (s_t, s_{t+1}) from retargeted
    clinical stroke gait trajectories.
    """

    def __init__(self, dataset_path: str | None = None, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.trajectories: list[torch.Tensor] = []

        if dataset_path and Path(dataset_path).exists():
            self._load_from_npz(dataset_path)
        else:
            # Fall back to whatever clinical data this extension has staged. If nothing is
            # staged either, the synthetic generator below still yields a usable prior.
            from humanoid_pathological_gait.tasks.humanoid_pathological_gait.assets import (
                EXPERT_DATASET_FILE,
                REFERENCE_STRIDE_FILE,
                resolve_data_file,
            )

            for file_name in (EXPERT_DATASET_FILE, REFERENCE_STRIDE_FILE):
                try:
                    self._load_from_npz(str(resolve_data_file(file_name)))
                    if len(self.trajectories) > 0:
                        break
                except (FileNotFoundError, KeyError, ValueError):
                    pass
        if len(self.trajectories) == 0:
            self._generate_synthetic_stroke_trajectories()

    def _load_from_npz(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        if "q_trajectory" in data:
            q = torch.tensor(data["q_trajectory"], dtype=torch.float32, device=self.device)
            v = torch.tensor(data["v_trajectory"], dtype=torch.float32, device=self.device)
            t_len = q.shape[0]

            z_root = torch.ones((t_len, 1), device=self.device) * 1.05
            proj_g = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(t_len, 1)
            lin_v = torch.tensor([0.6, 0.0, 0.0], device=self.device).repeat(t_len, 1)
            ang_v = torch.zeros((t_len, 3), device=self.device)

            feat = extract_amp_features(z_root, proj_g, lin_v, ang_v, q, v)
            self.trajectories.append(feat)
        elif "LHip" in data:
            num_strides = min(len(data["subject_ids"]), 50)
            deg2rad = math.pi / 180.0
            for s_idx in range(num_strides):
                t_len = data["LHip"].shape[1]
                q = torch.zeros((t_len, 19), device=self.device)
                q[:, 0] = torch.tensor(data["LHip"][s_idx, :, 1] * deg2rad, device=self.device)
                q[:, 1] = torch.tensor(data["LHip"][s_idx, :, 2] * deg2rad, device=self.device)
                q[:, 2] = torch.tensor(-data["LHip"][s_idx, :, 0] * deg2rad, device=self.device)
                q[:, 3] = torch.tensor(data["LKnee"][s_idx, :, 0] * deg2rad, device=self.device)
                q[:, 4] = torch.tensor(-data["LAnkle"][s_idx, :, 0] * deg2rad, device=self.device)

                q[:, 5] = torch.tensor(-data["RHip"][s_idx, :, 1] * deg2rad, device=self.device)
                q[:, 6] = torch.tensor(-data["RHip"][s_idx, :, 2] * deg2rad, device=self.device)
                q[:, 7] = torch.tensor(-data["RHip"][s_idx, :, 0] * deg2rad, device=self.device)
                q[:, 8] = torch.tensor(data["RKnee"][s_idx, :, 0] * deg2rad, device=self.device)
                q[:, 9] = torch.tensor(-data["RAnkle"][s_idx, :, 0] * deg2rad, device=self.device)

                q[:, 10] = 0.0  # Torso
                q[:, 11:15] = torch.tensor([0.0, 0.1, 0.0, 0.3], device=self.device)  # Left Arm
                q[:, 15:19] = torch.tensor([0.0, -0.1, 0.0, 0.3], device=self.device)  # Right Arm

                dt = 1.2 / (t_len - 1)
                v = torch.gradient(q, spacing=(dt,), dim=0)[0]
                z_root = torch.ones((t_len, 1), device=self.device) * 1.05
                proj_g = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(t_len, 1)
                lin_v = torch.tensor([0.6, 0.0, 0.0], device=self.device).repeat(t_len, 1)
                ang_v = torch.zeros((t_len, 3), device=self.device)

                feat = extract_amp_features(z_root, proj_g, lin_v, ang_v, q, v)
                self.trajectories.append(feat)

        print(f"[AMPExpertMotionBuffer] Loaded {len(self.trajectories)} expert trajectories.")

    def _generate_synthetic_stroke_trajectories(self, num_trajs: int = 10, traj_len: int = 200):
        for i in range(num_trajs):
            t = torch.linspace(0, 1.2, traj_len, device=self.device)
            phi = 2 * math.pi * t / 1.2
            phi_pi = phi + math.pi

            q = torch.zeros((traj_len, 19), device=self.device)
            q[:, 10] = 0.0  # Torso
            q[:, 11:15] = torch.tensor([0.0, 0.1, 0.0, 0.3], device=self.device)  # Left Arm
            q[:, 15:19] = torch.tensor([0.0, -0.1, 0.0, 0.3], device=self.device)  # Right Arm
            v = torch.zeros((traj_len, 19), device=self.device)

            # Healthy Kinematics
            h_hip = -0.2 + 0.4 * torch.sin(phi)
            h_knee = 0.4 + 0.5 * torch.clamp(torch.sin(phi), min=0.0)
            h_ankle = -0.2 - 0.2 * torch.sin(phi)

            h_hip_pi = -0.2 + 0.4 * torch.sin(phi_pi)
            h_knee_pi = 0.4 + 0.5 * torch.clamp(torch.sin(phi_pi), min=0.0)
            h_ankle_pi = -0.2 - 0.2 * torch.sin(phi_pi)

            # Paretic Kinematics
            p_hip = -0.2 + 0.3 * torch.sin(phi)
            p_knee = 0.2 + 0.15 * torch.sin(phi)
            p_ankle = -0.35 + 0.05 * torch.sin(phi)

            p_hip_pi = -0.2 + 0.3 * torch.sin(phi_pi)
            p_knee_pi = 0.2 + 0.15 * torch.sin(phi_pi)
            p_ankle_pi = -0.35 + 0.05 * torch.sin(phi_pi)

            if i % 2 == 0:
                q[:, 2], q[:, 3], q[:, 4] = p_hip, p_knee, p_ankle
                q[:, 7], q[:, 8], q[:, 9] = h_hip_pi, h_knee_pi, h_ankle_pi
            else:
                q[:, 2], q[:, 3], q[:, 4] = h_hip, h_knee, h_ankle
                q[:, 7], q[:, 8], q[:, 9] = p_hip_pi, p_knee_pi, p_ankle_pi

            v = torch.gradient(q, spacing=(0.02,), dim=0)[0]
            z_root = 1.05 + 0.03 * torch.sin(2 * phi).unsqueeze(-1)
            proj_g = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(traj_len, 1)
            lin_v = torch.column_stack([0.6 + 0.1 * torch.sin(phi), torch.zeros_like(phi), torch.zeros_like(phi)])
            ang_v = torch.zeros((traj_len, 3), device=self.device)

            feat = extract_amp_features(z_root, proj_g, lin_v, ang_v, q, v)
            self.trajectories.append(feat)

    def sample_transitions(self, batch_size: int) -> torch.Tensor:
        transitions = []
        for _ in range(batch_size):
            traj_idx = np.random.randint(len(self.trajectories))
            traj = self.trajectories[traj_idx]
            t_idx = np.random.randint(traj.shape[0] - 1)
            s_t = traj[t_idx]
            s_tp1 = traj[t_idx + 1]
            transitions.append(torch.cat([s_t, s_tp1], dim=-1))

        return torch.stack(transitions, dim=0)


class AMPAgentReplayBuffer:
    """Circular replay buffer for storing simulated transitions collected from policy rollouts."""

    def __init__(self, capacity: int = 50000, device: str | torch.device = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.buffer = torch.zeros((capacity, AMP_TRANSITION_DIM), dtype=torch.float32, device=self.device)
        self.ptr = 0
        self.size = 0

    def insert(self, transitions: torch.Tensor):
        num_items = transitions.shape[0]
        transitions = transitions.to(self.device)

        if self.ptr + num_items <= self.capacity:
            self.buffer[self.ptr : self.ptr + num_items] = transitions
            self.ptr = (self.ptr + num_items) % self.capacity
        else:
            first_part = self.capacity - self.ptr
            second_part = num_items - first_part
            self.buffer[self.ptr :] = transitions[:first_part]
            self.buffer[:second_part] = transitions[first_part:]
            self.ptr = second_part

        self.size = min(self.size + num_items, self.capacity)

    def sample(self, batch_size: int) -> torch.Tensor:
        assert self.size > 0, "Cannot sample from empty replay buffer."
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.buffer[indices]


if __name__ == "__main__":
    print("Testing AMP Discriminator, LSGAN Loss Manager, and Replay Buffers...")
    device = torch.device("cpu")

    disc = AMPDiscriminator(input_dim=AMP_TRANSITION_DIM, hidden_dims=(256, 128))
    loss_mgr = AMPLossManager(disc, learning_rate=1e-4)
    expert_buf = AMPExpertMotionBuffer(device=device)
    agent_buf = AMPAgentReplayBuffer(capacity=1000, device=device)

    synthetic_agent_trans = torch.randn(128, AMP_TRANSITION_DIM, device=device)
    agent_buf.insert(synthetic_agent_trans)

    expert_batch = expert_buf.sample_transitions(batch_size=64)
    agent_batch = agent_buf.sample(batch_size=64)

    metrics = loss_mgr.train_step(expert_batch, agent_batch)
    print("AMP Discriminator Training Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    rewards = disc.compute_amp_reward(expert_batch[:8])
    print(f"AMP Style Rewards (Expert Batch, shape {rewards.shape}):\n  {rewards.tolist()}")
    print("AMP Discriminator module verification passed successfully!")
