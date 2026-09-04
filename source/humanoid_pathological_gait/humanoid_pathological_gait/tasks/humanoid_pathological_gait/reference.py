# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-environment clinical reference gait state.

Holds the retargeted post-stroke stride and, for every environment, which side is
paretic and where in the stride that environment currently is. Sampling the stride
at an environment's phase gives the joint targets the action term drives toward and
the reward terms score against.

The stored stride was retargeted from a **left-paretic** subject. Environments
whose paretic side is the right one read a sagittally mirrored copy, so a single
policy learns both presentations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .h1_joints import NUM_JOINTS, H1JointLayout


class ReferenceGaitManager:
    """Reference stride playback and per-environment gait phase bookkeeping."""

    def __init__(
        self,
        stride_path: str | Path,
        layout: H1JointLayout,
        num_envs: int,
        device: torch.device | str,
        stride_duration_s: float = 1.2,
    ):
        self.layout = layout
        self.num_envs = num_envs
        self.device = torch.device(device)

        q_clinical, v_clinical, data_duration_s = self._load(Path(stride_path))
        self.stride_duration_s = float(data_duration_s if data_duration_s > 0.0 else stride_duration_s)

        # Store in simulation joint order so every consumer can compare elementwise
        # against ``robot.data.joint_pos`` without another permutation.
        self.ref_q = layout.to_sim_order(q_clinical.to(self.device))
        self.ref_v = layout.to_sim_order(v_clinical.to(self.device))
        self.ref_q_mirrored = layout.mirror(self.ref_q)
        self.ref_v_mirrored = layout.mirror(self.ref_v)
        self.num_samples = self.ref_q.shape[0]

        # Per-environment state.
        self.gait_phase = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.stride_duration = torch.full((num_envs,), self.stride_duration_s, dtype=torch.float32, device=self.device)
        self.paretic_side = torch.where(
            torch.rand(num_envs, device=self.device) > 0.5,
            torch.ones(num_envs, device=self.device),
            -torch.ones(num_envs, device=self.device),
        )

    def _load(self, path: Path) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Load the stride, deriving velocities by finite difference if absent."""
        data = np.load(str(path), allow_pickle=True)
        q = torch.tensor(np.asarray(data["q_trajectory"]), dtype=torch.float32)
        if q.ndim != 2 or q.shape[1] != NUM_JOINTS:
            raise ValueError(f"Reference stride at {path} has shape {tuple(q.shape)}, expected (T, {NUM_JOINTS}).")

        duration_s = 0.0
        if "time_vector" in data.files:
            time_vector = np.asarray(data["time_vector"])
            duration_s = float(time_vector[-1] - time_vector[0])

        if "v_trajectory" in data.files:
            v = torch.tensor(np.asarray(data["v_trajectory"]), dtype=torch.float32)
        else:
            dt = (duration_s if duration_s > 0.0 else 1.2) / max(q.shape[0] - 1, 1)
            v = torch.gradient(q, spacing=(dt,), dim=0)[0]

        return q, v, duration_s

    def advance(self, dt: float, rate_scale: float = 1.0) -> None:
        """Advance every environment's gait phase by ``dt`` seconds of stride time."""
        self.gait_phase = torch.remainder(self.gait_phase + rate_scale * dt / self.stride_duration, 1.0)

    def reset(self, env_ids: torch.Tensor, phase: torch.Tensor | float, paretic_side: torch.Tensor) -> None:
        """Reset gait phase and paretic side for the given environments."""
        self.gait_phase[env_ids] = (
            phase if isinstance(phase, torch.Tensor) else torch.full_like(self.gait_phase[env_ids], float(phase))
        )
        self.paretic_side[env_ids] = paretic_side

    def sample(self, env_ids: torch.Tensor | slice | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate reference joint positions and velocities at the current phases.

        Args:
            env_ids: Environments to sample, or ``None`` for all of them.

        Returns:
            ``(q_ref, v_ref)`` in simulation joint order, shape ``(len(env_ids), 19)``.
        """
        if env_ids is None:
            env_ids = slice(None)
        phase = self.gait_phase[env_ids]
        side = self.paretic_side[env_ids]

        # Linear interpolation between the two bracketing stride samples.
        position = phase * (self.num_samples - 1)
        lower = torch.floor(position).long().clamp_(0, self.num_samples - 1)
        upper = torch.clamp(lower + 1, max=self.num_samples - 1)
        blend = (position - lower.float()).unsqueeze(-1)

        def interpolate(table: torch.Tensor) -> torch.Tensor:
            return torch.lerp(table[lower], table[upper], blend)

        is_right_paretic = (side > 0).unsqueeze(-1)
        q_ref = torch.where(is_right_paretic, interpolate(self.ref_q_mirrored), interpolate(self.ref_q))
        v_ref = torch.where(is_right_paretic, interpolate(self.ref_v_mirrored), interpolate(self.ref_v))
        return q_ref, v_ref
