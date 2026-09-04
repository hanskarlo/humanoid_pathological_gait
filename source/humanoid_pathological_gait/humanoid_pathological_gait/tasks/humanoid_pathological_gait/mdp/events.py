# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms: reference-pose resets, asymmetric domain randomization, paretic pushes.

Domain randomization here is deliberately *asymmetric*. Randomizing the paretic limb
as widely as the sound one would wash out the fragile pathological limit cycle the
policy is meant to reproduce -- the deficit would just look like noise. So the sound
limb gets wide perturbation ranges (that is where robustness has to come from) while
the paretic limb is held to a couple of percent, and its actuators additionally carry
the reduced effort ceiling of hemiparetic weakness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..h1_pathological_env import H1PathologicalGaitEnv


def _uniform(shape: tuple[int, ...], value_range: tuple[float, float], device: torch.device) -> torch.Tensor:
    """Sample uniformly from ``value_range``."""
    low, high = value_range
    return torch.empty(shape, device=device).uniform_(low, high)


def _paretic_masks(env: H1PathologicalGaitEnv, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-joint ``(paretic, sound)`` leg masks for the given environments, shape ``(n, 19)``."""
    layout = env.joint_layout
    is_left_paretic = (env.reference_gait.paretic_side[env_ids] < 0).unsqueeze(-1)

    num = env_ids.shape[0]
    left = torch.zeros((num, len(layout.sim_names)), dtype=torch.bool, device=env.device)
    right = torch.zeros_like(left)
    left[:, layout.left_leg_ids] = True
    right[:, layout.right_leg_ids] = True

    paretic = torch.where(is_left_paretic, left, right)
    sound = torch.where(is_left_paretic, right, left)
    return paretic, sound


def reset_to_reference_pose(
    env: H1PathologicalGaitEnv,
    env_ids: torch.Tensor,
    randomize_paretic_side: bool = True,
    randomize_phase: bool = True,
    start_phase: float = 0.0,
    position_noise: float = 0.02,
    velocity_noise: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Draw a paretic side and gait phase, then place the robot on the reference pose.

    Resetting onto the reference rather than a fixed default pose is what makes phase
    conditioning meaningful: the observed phase and the body configuration agree from
    the first step of the episode.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    num_resets = env_ids.shape[0]

    if randomize_paretic_side:
        side = torch.where(
            torch.rand(num_resets, device=env.device) > 0.5,
            torch.ones(num_resets, device=env.device),
            -torch.ones(num_resets, device=env.device),
        )
    else:
        side = env.reference_gait.paretic_side[env_ids]

    phase = torch.rand(num_resets, device=env.device) if randomize_phase else float(start_phase)
    env.reference_gait.reset(env_ids, phase, side)

    q_ref, v_ref = env.reference_gait.sample(env_ids)
    q_ref = q_ref + position_noise * torch.randn_like(q_ref)
    v_ref = v_ref + velocity_noise * torch.randn_like(v_ref)

    limits = asset.data.soft_joint_pos_limits.torch[env_ids]
    q_ref = torch.clamp(q_ref, limits[..., 0], limits[..., 1])
    asset.write_joint_state_to_sim(q_ref, v_ref, env_ids=env_ids)


def randomize_asymmetric_joint_gains(
    env: H1PathologicalGaitEnv,
    env_ids: torch.Tensor,
    paretic_stiffness_range: tuple[float, float] = (0.98, 1.02),
    paretic_damping_range: tuple[float, float] = (0.98, 1.02),
    sound_stiffness_range: tuple[float, float] = (0.85, 1.15),
    sound_damping_range: tuple[float, float] = (0.70, 1.30),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Scale actuator stiffness and damping, tightly on the paretic leg and widely on the sound one."""
    asset: Articulation = env.scene[asset_cfg.name]
    paretic, sound = _paretic_masks(env, env_ids)
    shape = paretic.shape

    stiffness_scale = torch.ones(shape, device=env.device)
    damping_scale = torch.ones(shape, device=env.device)
    stiffness_scale = torch.where(paretic, _uniform(shape, paretic_stiffness_range, env.device), stiffness_scale)
    stiffness_scale = torch.where(sound, _uniform(shape, sound_stiffness_range, env.device), stiffness_scale)
    damping_scale = torch.where(paretic, _uniform(shape, paretic_damping_range, env.device), damping_scale)
    damping_scale = torch.where(sound, _uniform(shape, sound_damping_range, env.device), damping_scale)

    asset.write_joint_stiffness_to_sim_index(
        stiffness=env.default_joint_stiffness[env_ids] * stiffness_scale, env_ids=env_ids
    )
    asset.write_joint_damping_to_sim_index(damping=env.default_joint_damping[env_ids] * damping_scale, env_ids=env_ids)


def randomize_asymmetric_effort_limits(
    env: H1PathologicalGaitEnv,
    env_ids: torch.Tensor,
    paretic_effort_scale: float = 0.4,
    paretic_range: tuple[float, float] = (1.0, 1.0),
    sound_range: tuple[float, float] = (0.85, 1.15),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Apply hemiparetic weakness: cut the paretic leg's torque ceiling, jitter the sound leg's.

    ``paretic_effort_scale`` is the fraction of nominal torque the paretic leg can still
    produce -- 0.4 puts the H1's 300 Nm leg actuators near the 120 Nm figure used for the
    paretic side in the clinical model, and its 100 Nm ankle near 40 Nm.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    paretic, sound = _paretic_masks(env, env_ids)
    shape = paretic.shape

    scale = torch.ones(shape, device=env.device)
    scale = torch.where(paretic, paretic_effort_scale * _uniform(shape, paretic_range, env.device), scale)
    scale = torch.where(sound, _uniform(shape, sound_range, env.device), scale)

    asset.write_joint_effort_limit_to_sim_index(
        limits=env.default_joint_effort_limits[env_ids] * scale, env_ids=env_ids
    )


def randomize_asymmetric_leg_mass(
    env: H1PathologicalGaitEnv,
    env_ids: torch.Tensor,
    paretic_range: tuple[float, float] = (0.98, 1.02),
    sound_range: tuple[float, float] = (0.80, 1.25),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Scale leg link masses, tightly on the paretic side and widely on the sound side."""
    asset: Articulation = env.scene[asset_cfg.name]
    is_left_paretic = (env.reference_gait.paretic_side[env_ids] < 0).unsqueeze(-1)
    num = env_ids.shape[0]

    left_ids = env.left_leg_body_ids
    right_ids = env.right_leg_body_ids
    scale = torch.ones((num, asset.num_bodies), device=env.device)

    paretic_scale = _uniform((num, len(left_ids)), paretic_range, env.device)
    sound_scale = _uniform((num, len(left_ids)), sound_range, env.device)
    scale[:, left_ids] = torch.where(is_left_paretic, paretic_scale, sound_scale)
    scale[:, right_ids] = torch.where(is_left_paretic, sound_scale, paretic_scale)

    asset.set_masses_index(masses=env.default_body_mass[env_ids] * scale, env_ids=env_ids)


def push_biased_toward_paretic_side(
    env: H1PathologicalGaitEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    paretic_bias: float = 1.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Push the base, amplifying pushes that fall toward the paretic side.

    Post-stroke balance fails asymmetrically: a perturbation toward the weak side is
    much harder to arrest than the mirror-image one. The push is sampled in the robot's
    own yaw frame -- so "lateral" means lateral to the walking direction, not to the
    world axes -- scaled by ``paretic_bias`` when it points at the paretic side, then
    rotated back into world coordinates and added to the current base velocity.

    Args:
        env: The environment.
        env_ids: Environments to push.
        velocity_range: Per-axis push ranges (m/s for ``x``/``y``/``z``, rad/s for
            ``roll``/``pitch``/``yaw``), in the robot's yaw frame.
        paretic_bias: Multiplier on the lateral push when it points at the paretic side.
        asset_cfg: The articulation to push.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    num = env_ids.shape[0]

    delta = torch.zeros((num, 6), device=env.device)
    for axis, index in (("x", 0), ("y", 1), ("z", 2), ("roll", 3), ("pitch", 4), ("yaw", 5)):
        if axis in velocity_range:
            delta[:, index] = _uniform((num,), velocity_range[axis], env.device)

    # paretic_side is -1 for left; +y in the robot frame points left, so a push points at
    # the paretic side when lateral velocity and -paretic_side share a sign.
    lateral = delta[:, 1]
    toward_paretic = (lateral * -env.reference_gait.paretic_side[env_ids]) > 0
    delta[:, 1] = torch.where(toward_paretic, lateral * paretic_bias, lateral)

    heading_quat = yaw_quat(asset.data.root_link_quat_w.torch[env_ids])
    delta_w = torch.cat(
        [quat_apply(heading_quat, delta[:, :3]), quat_apply(heading_quat, delta[:, 3:])],
        dim=-1,
    )
    velocity = asset.data.root_com_vel_w.torch[env_ids] + delta_w
    asset.write_root_com_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
