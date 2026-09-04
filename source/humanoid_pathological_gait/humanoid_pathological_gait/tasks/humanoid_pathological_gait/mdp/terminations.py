# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific termination terms.

Falls and time-outs are covered by stock Isaac Lab terms. What an imitation task
additionally wants is to stop an episode once the robot has diverged far enough
from the reference that the remaining rollout carries no useful learning signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..h1_pathological_env import H1PathologicalGaitEnv


def reference_tracking_divergence(
    env: H1PathologicalGaitEnv,
    max_rms_error: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the clinically weighted RMS joint tracking error exceeds a bound.

    Args:
        env: The environment.
        max_rms_error: Weighted RMS joint position error (rad) at which to give up.
        asset_cfg: The articulation being tracked.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q_ref, _ = env.reference_gait.sample()
    weights = env.joint_layout.tracking_weights
    weighted_mse = torch.sum(torch.square(asset.data.joint_pos.torch - q_ref) * weights, dim=-1) / torch.sum(weights)
    return torch.sqrt(weighted_mse) > max_rms_error
