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

import math

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
    target_velocity: float | None = None,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """RBF reward on forward walking speed in the robot's own frame.

    ``target_velocity=None`` (the default) takes the speed the **reference stride itself
    travels at**, from the floating base the retargeter solved. That is the honest target:
    the previous hard-coded 0.5 m/s was twice the reference's own 0.244 m/s, and since joint
    tracking outweighs this term 15.0 to 2.0, the two objectives simply fought -- the policy
    settled near the reference speed and collected a permanent shortfall here. Post-stroke
    gait is slow; asking for a speed the reference does not contain asks the policy to stop
    tracking it.

    Pass a float to override, e.g. to study speed as an independent variable.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    forward_speed = asset.data.root_lin_vel_b.torch[:, 0]
    if target_velocity is None:
        target_velocity = env.reference_gait.reference_speed
        if not math.isfinite(target_velocity):
            raise ValueError(
                "target_velocity=None needs reference_speed_ms in the reference archive; "
                "regenerate it with data/batch_parse_gait.py --solver gmr --source measured."
            )
    return torch.exp(-torch.square(forward_speed - target_velocity) / std**2)


def swing_timing(
    env: H1PathologicalGaitEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    paretic_weight: float = 2.0,
    max_class_weight: float = 4.0,
) -> torch.Tensor:
    """Reward each foot being loaded (or not) when the reference says it should be.

    This is the term the 2026-09-05 baseline was missing, and its absence is why two of
    three seeds shuffled: nothing asked a foot to leave the ground. ``paretic_foot_clearance``
    is gated on the foot *already* being airborne, so a policy that plants the paretic foot
    forgoes that term and is otherwise unpenalised -- both feet down is a local optimum, and
    it was found. Measured double support ran to 0.82-0.89 of the cycle against 0.37 in this
    cohort's own patients.

    The reference schedule comes from ``reference_contact`` in the stride archive: each
    limb's swing *fraction* is measured from its own toe-off events, and the retargeted foot
    kinematics decide *when* within the cycle (see
    ``kinematics.gmr_solver.contact_schedule`` for why neither source suffices alone).

    Scores agreement per foot in the paretic/sound frame, weighting the paretic foot more
    since its swing is the deficit under study. Returns 0 when the archive carries no
    schedule, so an older reference degrades to the previous behaviour rather than erroring.

    **Class-balanced, and it has to be.** The schedule says "down" 61% of the cycle for the
    paretic foot and 84% for the sound one, so plain agreement hands a policy that never
    lifts either foot **0.683 of the maximum for free**, leaving only 0.317 contestable
    against joint tracking at weight 15.0. The first full-scale run with the unbalanced form
    ended at 0.635 -- *below* the do-nothing floor -- and double support went from 0.820 to
    0.837, slightly worse than the baseline it was meant to fix.

    Each frame is therefore weighted by the inverse frequency of its own class, so stance
    frames and swing frames contribute equally in expectation. A constant policy -- always
    down or always up -- scores exactly 0.5, a perfect one 1.0, and the whole upper half is
    contestable. The per-frame weight is capped (see ``max_class_weight``) because a limb
    with a short swing would otherwise produce large single-step spikes.
    """
    schedule = env.reference_gait.sample_contact()
    if schedule is None:
        return torch.zeros(env.num_envs, device=env.device)

    sensor: ContactSensor = env.scene[sensor_cfg.name]
    force = torch.norm(sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids], dim=-1)
    loaded = (force > contact_threshold).float()

    # body_ids are ordered (left, right); the schedule is (paretic, sound), so swap the
    # robot's columns for right-paretic environments before comparing.
    is_right_paretic = (env.reference_gait.paretic_side > 0).unsqueeze(-1)
    loaded = torch.where(is_right_paretic, loaded.flip(-1), loaded)

    agreement = 1.0 - torch.abs(loaded - schedule)

    # Inverse-frequency class balancing. `stance` is each foot's share of the cycle spent
    # down, so a stance frame is worth 1/(2*stance) and a swing frame 1/(2*(1-stance)):
    # each class then contributes 0.5 in expectation and a constant policy scores 0.5.
    stance = env.reference_gait.contact_stance_fraction.clamp(1e-3, 1.0 - 1e-3)
    stance_weight = (0.5 / stance).clamp(max=max_class_weight)
    swing_weight = (0.5 / (1.0 - stance)).clamp(max=max_class_weight)
    class_weight = torch.where(schedule > 0.5, stance_weight, swing_weight)

    foot_weight = torch.tensor([paretic_weight, 1.0], device=env.device)
    return (agreement * class_weight * foot_weight).sum(dim=-1) / foot_weight.sum()


def track_root_progression(
    env: H1PathologicalGaitEnv,
    std: float = 0.15,
    cross_track_weight: float = 0.5,
) -> torch.Tensor:
    """Reward the root being where the reference's root is, at the phase the reference is at.

    The joint reference says which pose to hold at each phase; it says nothing about where
    the robot should *be*. A policy can satisfy it while standing still, which the
    2026-09-05 baseline did, or while taking three short steps per cycle instead of one long
    one, which the class-balanced ``swing_timing`` run did -- 3.58 stance periods per cycle
    and a 0.185 m stride against the reference's 0.447 m, at roughly three times its
    cadence. Both are the same underlying freedom: nothing tied position to phase.

    The reference's own floating base, solved by the retargeter and stored in the archive,
    states the constraint directly. Error is measured from the start of the current gait
    cycle in the heading the environment had then, so it is progress *within* a stride
    rather than accumulated drift, and it is invariant to which way the robot is facing.

    Along-track error is what carries cadence and stride length; cross-track is weighted
    lower because veering is already penalised by the heading and orientation terms.
    """
    error = env.root_progression_error()
    if error is None:
        return torch.zeros(env.num_envs, device=env.device)
    weighted = torch.square(error[:, 0]) + cross_track_weight * torch.square(error[:, 1])
    return torch.exp(-weighted / std**2)


def track_base_height(
    env: H1PathologicalGaitEnv,
    target_height: float = 1.05,
    std: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """RBF reward keeping the pelvis near its nominal standing height.

    ``std`` has to be read against foot clearance, not against how much pelvis travel
    looks reasonable on its own. The reference stride clears its paretic foot by only
    68 mm, so every millimetre the pelvis drops comes straight out of that budget: a
    crouch of 46 mm leaves 22 mm, and the foot scuffs through what should be swing.
    A permissive ``std`` therefore buys a fragmented contact pattern rather than a
    softer posture constraint, which is why this defaults tight.
    """
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


#: Lateral width of one H1 foot's contact patch, in metres.
#:
#: Measured on the live Isaac stage, from the collision mesh PhysX actually uses: the
#: ``left_ankle_link`` collider spans y = +0.1629..+0.2429 about a link origin at
#: y = +0.2029, i.e. **0.0800 m wide and exactly centred on the link origin**. That the
#: offset is zero is what licenses using ``body_link_pos_w`` as the foot centre in
#: :func:`compute_xcom_and_mos`; it had been an unstated assumption.
#:
#: Getting this took three attempts and the first two were wrong, so: the asset is
#: *instanceable*, and its colliders live in USD prototypes that a default ``Usd.PrimRange``
#: skips -- both the distributed layer and the live stage report no geometry at all under
#: the ankle unless traversed with ``Usd.TraverseInstanceProxies()``. Reading the
#: ``mujoco_menagerie`` H1 instead gives 0.088 m, which is the right order but the wrong
#: robot; its collision hull is not the one Isaac collides with (its foot is also 0.176 m
#: long against this asset's 0.240 m).
#:
#: The value used before all this was 0.12 m -- 50% too wide, inflating every margin by
#: exactly half the difference, 20 mm, against a true median margin of about 22 mm.
H1_FOOT_WIDTH_M = 0.080


def compute_xcom_and_mos(
    env: H1PathologicalGaitEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    foot_width: float = H1_FOOT_WIDTH_M,
    contact_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extrapolated centre of mass and mediolateral margin of stability.

    Everything is expressed in the robot's yaw frame, so "lateral" stays lateral
    however the robot is heading. The support polygon spans the feet whose contact force
    exceeds ``contact_threshold``.

    **With no foot loaded there is no base of support and Hof's margin is undefined**, so
    a convention is needed. This used to fall back on the polygon spanned by both feet, on
    the reasoning that it "yields the negative margin a flight phase deserves". Measured
    over a 51,200-sample rollout, it does the opposite: 3.9% of samples have no foot
    loaded, and 89.8% of those report a *positive* margin, median +0.183 m against an
    overall median of +0.072 m. The feet are furthest apart in mid-swing, so the fabricated
    polygon is widest exactly when nothing is supporting it, and the stability reward pays
    out in full for leaving the ground.

    The convention here instead reports how far the centre of mass sits from the nearest
    foot it could land on, negated -- zero when the XCoM is still over a foot's patch,
    increasingly negative as it escapes. It is never positive, it is continuous with the
    supported case at the contact transition, and it is a limiting-case convention rather
    than a Hof margin. Callers reporting MoS statistics should exclude these samples or
    state their share; ``scripts/evaluate.py`` reports the share.

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
    any_contact = in_contact.any(dim=-1)

    foot_y = foot_pos[..., 1]
    half_width = 0.5 * foot_width
    lower_edge = torch.where(in_contact, foot_y - half_width, torch.full_like(foot_y, float("inf")))
    upper_edge = torch.where(in_contact, foot_y + half_width, torch.full_like(foot_y, float("-inf")))
    supported = torch.minimum(
        upper_edge.max(dim=-1).values - xcom[:, 1],
        xcom[:, 1] - lower_edge.min(dim=-1).values,
    )

    # Flight: no base of support, so report the (negated) lateral distance to the nearest
    # foot's patch instead. Never positive; zero while the XCoM is still over a foot.
    escape = torch.clamp((xcom[:, 1].unsqueeze(-1) - foot_y).abs() - half_width, min=0.0)
    airborne = -escape.min(dim=-1).values

    mos_lateral = torch.where(any_contact, supported, airborne)
    return xcom, mos_lateral, in_contact


def margin_of_stability(
    env: H1PathologicalGaitEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_margin: float = 0.04,
    std: float = 0.05,
    foot_width: float = H1_FOOT_WIDTH_M,
    tipping_penalty_scale: float = 5.0,
) -> torch.Tensor:
    """Reward a mediolateral margin of stability at or above ``target_margin``.

    Margins beyond the target score the full reward; shortfalls decay as a Gaussian,
    and a margin that goes negative -- the XCoM outside the support polygon, i.e. the
    robot committed to a fall it cannot arrest without a step -- is penalised quadratically.

    ``target_margin`` is a *true* margin. It was calibrated while ``foot_width`` was 0.12 m,
    which inflated every margin by 20 mm, so the 0.04 m target was really asking for about
    0.020 m; with the measured width it now asks for what it says. Expect this term to be
    harder to satisfy than it was, and any policy trained before this to have been scored
    against a target half as demanding.
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
