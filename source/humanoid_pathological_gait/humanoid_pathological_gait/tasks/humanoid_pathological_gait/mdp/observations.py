# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific observation terms: the clinical reference and the gait clock.

The proprioceptive half of the observation comes from stock Isaac Lab terms. What
the policy additionally needs is the target it is imitating (reference joint
positions and velocities), which side is paretic, and where it is in the stride.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..h1_pathological_env import H1PathologicalGaitEnv


def reference_joint_pos(env: H1PathologicalGaitEnv) -> torch.Tensor:
    """Reference joint positions at the current gait phase. Shape ``(num_envs, 19)``."""
    q_ref, _ = env.reference_gait.sample()
    return q_ref


def reference_joint_vel(env: H1PathologicalGaitEnv) -> torch.Tensor:
    """Reference joint velocities at the current gait phase. Shape ``(num_envs, 19)``."""
    _, v_ref = env.reference_gait.sample()
    return v_ref


def reference_joint_pos_error(env: H1PathologicalGaitEnv) -> torch.Tensor:
    """Signed tracking error against the reference pose. Shape ``(num_envs, 19)``."""
    q_ref, _ = env.reference_gait.sample()
    return env.scene["robot"].data.joint_pos.torch - q_ref


def paretic_side(env: H1PathologicalGaitEnv) -> torch.Tensor:
    """Which side is paretic: ``-1`` left, ``+1`` right. Shape ``(num_envs, 1)``."""
    return env.reference_gait.paretic_side.unsqueeze(-1)


def gait_phase(env: H1PathologicalGaitEnv) -> torch.Tensor:
    """Gait phase as a ``(sin, cos)`` pair so it stays continuous across the stride wrap."""
    angle = 2.0 * math.pi * env.reference_gait.gait_phase
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
