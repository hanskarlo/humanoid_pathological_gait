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
from isaaclab.utils.math import quat_apply, quat_mul, quat_from_euler_xyz, yaw_quat

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
    scatter_xy: float = 0.5,
    randomize_yaw: bool = True,
    root_lin_vel_noise: float = 0.1,
    root_ang_vel_noise: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Draw a paretic side and gait phase, then place the **whole** robot on the reference.

    Resetting onto the reference rather than a fixed default pose is what makes phase
    conditioning meaningful: the observed phase and the body configuration agree from the
    first step of the episode.

    That agreement used to stop at the joints. This term wrote ``q_ref``/``v_ref`` and a
    separate ``reset_root_state_uniform`` placed the floating base at its default pose with
    a zero-mean velocity, so every episode began in a state the reference never occupies:

    * **Linear velocity zero-mean, against the reference's +0.244 m/s** (0.092 to 0.435 over
      the stride). The legs were placed mid-swing with reference joint velocities while the
      body they hang from had no momentum. Walking single support is a controlled fall
      forward onto the swing foot; from rest there is nothing to fall *into*, and the only
      way not to topple sideways is to plant the second foot. 55.8% of the reference stride
      is single support, so more than half of all reset draws landed in exactly that state.
    * **Pelvis level, against a reference carrying +1.78 to +9.65 deg of coronal roll.** That
      obliquity is the mechanism this stride uses for lateral foot clearance, and pelvic
      hiking is one of the hallmarks the project set out to reproduce. It was being zeroed
      at every reset.
    * **Height from the articulation default (1.05 m)** rather than the reference's
      phase-dependent 1.0364-1.0664. Small next to the other two, and fixed for free.

    This is the reference-state-initialisation of Peng et al. (DeepMimic, 2018) done in
    full: the point of RSI is that the policy gets gradient from states it could not reach
    by exploration, and that only holds if the state it is placed in is the reference's.
    A pose without its momentum is not a noisy sample of the reference, it is a different
    state, and no amount of reward shaping reaches the one that was skipped.

    What is still randomised, because it has to be: planar position, heading, and small
    noise on top of every reference quantity. Falls back to the previous behaviour, default
    root pose included, for a stride archive predating ``root_quaternion``.
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

    _reset_root_to_reference(
        env,
        env_ids,
        asset,
        scatter_xy=scatter_xy,
        randomize_yaw=randomize_yaw,
        lin_vel_noise=root_lin_vel_noise,
        ang_vel_noise=root_ang_vel_noise,
    )


def _reset_root_to_reference(
    env: H1PathologicalGaitEnv,
    env_ids: torch.Tensor,
    asset: Articulation,
    scatter_xy: float,
    randomize_yaw: bool,
    lin_vel_noise: float,
    ang_vel_noise: float,
) -> None:
    """Place the floating base on the reference root state, at a random heading.

    The reference tables are stored with the stride's own heading removed, so the episode's
    yaw is composed back on here and the reference's linear and angular velocities are
    rotated into it. Rotating the velocities is not optional: a robot facing 90 deg off the
    capture heading and given the capture's forward velocity would be launched sideways,
    which is a worse initial state than the one this replaces.
    """
    num_resets = env_ids.shape[0]
    device = env.device
    root_state = env.reference_gait.sample_root_state(env_ids)

    default_pose = asset.data.default_root_pose.torch[env_ids].clone()
    default_velocity = asset.data.default_root_vel.torch[env_ids].clone()

    zeros = torch.zeros(num_resets, device=device)
    yaw = (
        torch.empty(num_resets, device=device).uniform_(-torch.pi, torch.pi) if randomize_yaw else zeros
    )
    yaw_rotation = quat_from_euler_xyz(zeros, zeros, yaw)

    position = default_pose[:, 0:3] + env.scene.env_origins[env_ids]
    if scatter_xy > 0.0:
        position[:, :2] += torch.empty((num_resets, 2), device=device).uniform_(-scatter_xy, scatter_xy)

    if root_state is None:
        # Archive predates ``root_quaternion``: keep the old default-pose behaviour rather
        # than fail, but the reference-state initialisation above is then only partial.
        orientation = quat_mul(default_pose[:, 3:7], yaw_rotation)
        velocity = default_velocity
    else:
        height, quaternion, linear_velocity, angular_velocity = root_state
        # The reference height is measured from the ground; env origins carry the ground.
        position[:, 2] = env.scene.env_origins[env_ids][:, 2] + height
        orientation = quat_mul(yaw_rotation, quaternion)
        velocity = torch.cat(
            [quat_apply(yaw_rotation, linear_velocity), quat_apply(yaw_rotation, angular_velocity)],
            dim=-1,
        )

    if lin_vel_noise > 0.0:
        velocity[:, 0:3] += lin_vel_noise * torch.randn_like(velocity[:, 0:3])
    if ang_vel_noise > 0.0:
        velocity[:, 3:6] += ang_vel_noise * torch.randn_like(velocity[:, 3:6])

    asset.write_root_pose_to_sim_index(
        root_pose=torch.cat([position, orientation], dim=-1), env_ids=env_ids
    )
    asset.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)


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
