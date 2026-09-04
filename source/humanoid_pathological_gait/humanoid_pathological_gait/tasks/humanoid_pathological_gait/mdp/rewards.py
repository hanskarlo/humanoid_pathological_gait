# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for pathological gait imitation and dynamic balance.

Two families live here:

* **Kinematic tracking** -- radial-basis kernels on joint position and velocity
  error, weighted per joint so ankles and knees (where the stroke deficits show)
  dominate the score.
* **Dynamic balance** -- the Extrapolated Centre of Mass (XCoM) and mediolateral
  Margin of Stability (MoS) of Hof et al. (2005). Unlike the analytical predecessor,
  which approximated foot placement from the root pose and assumed permanent double
  support, these read the real whole-body centre of mass and build the support
  polygon from the feet actually in contact this step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor

    from ..h1_pathological_env import H1PathologicalGaitEnv

GRAVITY = 9.81


def joint_pos_tracking(
    env: H1PathologicalGaitEnv, std: float = 0.35, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Clinically weighted RBF reward on joint position tracking error, in ``[0, 1]``."""
    asset: Articulation = env.scene[asset_cfg.name]
    q_ref, _ = env.reference_gait.sample()
    error_sq = torch.square(asset.data.joint_pos.torch - q_ref)
    weights = env.joint_layout.tracking_weights
    return torch.sum(torch.exp(-error_sq / std**2) * weights, dim=-1) / torch.sum(weights)


def joint_vel_tracking(
    env: H1PathologicalGaitEnv, std: float = 2.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Clinically weighted RBF reward on joint velocity tracking error, in ``[0, 1]``."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, v_ref = env.reference_gait.sample()
    error_sq = torch.square(asset.data.joint_vel.torch - v_ref)
    weights = env.joint_layout.tracking_weights
    return torch.sum(torch.exp(-error_sq / std**2) * weights, dim=-1) / torch.sum(weights)


def track_forward_velocity(
    env: H1PathologicalGaitEnv,
    target_velocity: float = 0.5,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """RBF reward on forward walking speed in the robot's own frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    forward_speed = asset.data.root_lin_vel_b.torch[:, 0]
    return torch.exp(-torch.square(forward_speed - target_velocity) / std**2)


def track_base_height(
    env: H1PathologicalGaitEnv,
    target_height: float = 1.05,
    std: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """RBF reward keeping the pelvis near its nominal standing height."""
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w.torch[:, 2]
    return torch.exp(-torch.square(height - target_height) / std**2)


def _whole_body_com(asset: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
    """Mass-weighted centre of mass position and velocity in world coordinates."""
    mass = asset.data.body_mass.torch.unsqueeze(-1)
    total_mass = mass.sum(dim=1)
    com_pos = (asset.data.body_com_pos_w.torch * mass).sum(dim=1) / total_mass
    com_vel = (asset.data.body_com_lin_vel_w.torch * mass).sum(dim=1) / total_mass
    return com_pos, com_vel


def compute_xcom_and_mos(
    env: H1PathologicalGaitEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    foot_width: float = 0.12,
    contact_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extrapolated centre of mass and mediolateral margin of stability.

    Everything is expressed in the robot's yaw frame, so "lateral" stays lateral
    however the robot is heading. The support polygon spans the feet whose contact
    force exceeds ``contact_threshold``; in flight, both feet define it, which yields
    the negative margin that a flight phase deserves.

    Args:
        env: The environment.
        asset_cfg: The articulation, with ``body_names`` resolving to the two feet.
        sensor_cfg: The contact sensor, with ``body_ids`` matching the same two feet.
        foot_width: Lateral width of a foot's contact patch (m).
        contact_threshold: Net contact force (N) above which a foot counts as loaded.

    Returns:
        ``(xcom, mos_lateral, in_contact)`` -- the XCoM in the yaw frame (shape
        ``(num_envs, 2)``), the lateral margin in metres (shape ``(num_envs,)``, negative
        when the XCoM has left the support polygon), and a per-foot contact mask.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_cfg.name]

    com_pos_w, com_vel_w = _whole_body_com(asset)
    foot_pos_w = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids]

    # Rotate into the robot's yaw frame, taking the root position as the origin.
    # Quaternions in this Isaac Lab release are (x, y, z, w).
    heading_quat = yaw_quat(asset.data.root_link_quat_w.torch)
    root_pos_w = asset.data.root_link_pos_w.torch
    com_pos = quat_apply_inverse(heading_quat, com_pos_w - root_pos_w)
    com_vel = quat_apply_inverse(heading_quat, com_vel_w)
    # quat_apply_inverse does not broadcast, so expand the heading over the two feet.
    num_feet = foot_pos_w.shape[1]
    foot_quat = heading_quat.unsqueeze(1).expand(-1, num_feet, -1)
    foot_pos = quat_apply_inverse(foot_quat, foot_pos_w - root_pos_w.unsqueeze(1))

    # XCoM = CoM + CoM velocity / omega_0, with omega_0 the inverted-pendulum frequency.
    # Clamp the pendulum length so a collapsing robot cannot produce a runaway omega_0.
    com_height = torch.clamp(com_pos_w[:, 2], min=0.3, max=1.3)
    omega_0 = torch.sqrt(GRAVITY / com_height).unsqueeze(-1)
    xcom = com_pos[:, :2] + com_vel[:, :2] / omega_0

    # Support polygon: lateral extent of the loaded feet, widened by half a foot each side.
    net_force = torch.norm(sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids], dim=-1)
    in_contact = net_force > contact_threshold
    any_contact = in_contact.any(dim=-1, keepdim=True)
    # In flight no foot is loaded; fall back to both so the margin still measures how far
    # the XCoM sits outside where the feet are, rather than going undefined.
    weights = torch.where(any_contact, in_contact, torch.ones_like(in_contact))

    foot_y = foot_pos[..., 1]
    lower_edge = torch.where(weights, foot_y - 0.5 * foot_width, torch.full_like(foot_y, float("inf")))
    upper_edge = torch.where(weights, foot_y + 0.5 * foot_width, torch.full_like(foot_y, float("-inf")))
    mos_lateral = torch.minimum(
        upper_edge.max(dim=-1).values - xcom[:, 1],
        xcom[:, 1] - lower_edge.min(dim=-1).values,
    )
    return xcom, mos_lateral, in_contact


def margin_of_stability(
    env: H1PathologicalGaitEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_margin: float = 0.04,
    std: float = 0.05,
    foot_width: float = 0.12,
    tipping_penalty_scale: float = 5.0,
) -> torch.Tensor:
    """Reward a mediolateral margin of stability at or above ``target_margin``.

    Margins beyond the target score the full reward; shortfalls decay as a Gaussian,
    and a margin that goes negative -- the XCoM outside the support polygon, i.e. the
    robot committed to a fall it cannot arrest without a step -- is penalised quadratically.
    """
    _, mos_lateral, _ = compute_xcom_and_mos(env, asset_cfg, sensor_cfg, foot_width=foot_width)
    shortfall = torch.clamp(target_margin - mos_lateral, min=0.0)
    tipping = torch.square(torch.clamp(-mos_lateral, min=0.0))
    return torch.exp(-torch.square(shortfall) / std**2) - tipping_penalty_scale * tipping


def paretic_foot_clearance(
    env: H1PathologicalGaitEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.10,
    std: float = 0.04,
) -> torch.Tensor:
    """Reward swing-phase clearance of the paretic foot.

    Foot drop is the deficit this task exists to reproduce, and the reference stride
    alone does not tell the policy when the paretic foot is meant to be off the ground.
    The term is active only while that foot is unloaded, so it shapes swing without
    fighting stance.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_cfg.name]

    foot_height = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids, 2]
    net_force = torch.norm(sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids], dim=-1)

    # body_ids are ordered (left, right); pick the paretic one per environment.
    is_right_paretic = (env.reference_gait.paretic_side > 0).long()
    index = is_right_paretic.unsqueeze(-1)
    paretic_height = torch.gather(foot_height, 1, index).squeeze(-1)
    paretic_force = torch.gather(net_force, 1, index).squeeze(-1)

    in_swing = (paretic_force <= 1.0).float()
    return torch.exp(-torch.square(paretic_height - target_height) / std**2) * in_swing


def spastic_torque_l2(env: H1PathologicalGaitEnv) -> torch.Tensor:
    """Squared TSRT reflex torque, as a diagnostic of how hard the policy fights spasticity."""
    return torch.sum(torch.square(env.applied_spastic_torque), dim=-1)
