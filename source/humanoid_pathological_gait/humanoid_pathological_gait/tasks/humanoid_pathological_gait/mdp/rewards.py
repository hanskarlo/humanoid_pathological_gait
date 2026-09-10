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
    env: H1PathologicalGaitEnv,
    std: float = 0.35,
    rom_scale: float | None = None,
    min_std: float = 0.05,
    max_std: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Clinically weighted RBF reward on joint position tracking error, in ``[0, 1]``.

    With ``rom_scale=None`` this is the original single shared width. **That form could not
    see the pathology.** The reward is ``exp(-e^2/std^2)`` per joint, so at ``std = 0.35`` rad
    it falls to half at 16.7 deg of error -- while 16 of the reference stride's 19 joints have
    a total range of motion *below* that. A policy that froze every joint at its reference
    mean and never moved scored **0.9106** of this term, which carries weight 15.0, the
    largest in the reward. Only the sound limb's knee (ROM 50.6 deg) and hip pitch (47.5 deg)
    had real contestable range; 17 of 19 joints paid out over 0.90 for doing nothing.

    That is not a tuning detail, it is the shape of every result this project has produced.
    The paretic limb is where the small ranges are -- knee 18.4, ankle 6.7, hip roll 5.5 deg --
    so the paretic limb was the limb the objective could least resolve. It explains the
    inverted weight-bearing asymmetry (the paretic side was nearly free), the hip roll sitting
    against its mechanical stop for 62-75% of the gait (a 30 deg error there costs about 4% of
    the achieved reward), and it means "foot drop reproduces" needs re-examining: a frozen
    ankle and a dropped foot are indistinguishable under a 16.7 deg-wide RBF.

    With ``rom_scale`` set, each joint gets ``std_j = clamp(rom_scale * ROM_j, min_std,
    max_std)`` from :meth:`ReferenceGaitManager.joint_rom`, so the term has comparable
    resolving power on a 5 deg joint and a 50 deg one. ``max_std`` never *loosens* a joint
    beyond the original width; ``min_std`` keeps a gradient far from the target, which is the
    failure mode the ``base_height`` narrowing hit when a sharpened RBF stopped shaping at all.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q_ref, _ = env.reference_gait.sample()
    error_sq = torch.square(asset.data.joint_pos.torch - q_ref)
    weights = env.joint_layout.tracking_weights

    if rom_scale is None:
        width_sq = std**2
    else:
        width_sq = torch.square(
            torch.clamp(rom_scale * env.reference_gait.joint_rom(), min=min_std, max=max_std)
        )
    return torch.sum(torch.exp(-error_sq / width_sq) * weights, dim=-1) / torch.sum(weights)


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
    target_height: float | None = None,
    std: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """RBF reward tracking the reference pelvis height at the current gait phase.

    ``target_height=None`` (the default) follows the reference's own pelvis trajectory.
    That matters because the pelvis is not level while walking: it rises and falls 30.0 mm
    peak to peak over the stride. Tracking a *constant* instead -- which this term used to
    do -- asks the policy to hold still vertically while every other term asks it to walk,
    and caps the score at whatever the reference's own bob costs.

    Pass a float to pin a constant target; the reward also falls back to 1.05 m when the
    stride archive predates ``root_translation`` and carries no height to track.

    On ``std``: narrowing it is not the same as weighting it more. Going 0.15 -> 0.04 to
    charge harder for crouching was measured over a matched run and made every gait metric
    worse, the pelvis height it was aimed at included -- past ~80 mm of error the Gaussian
    is flat, so once the policy left the spike nothing pulled it back and the term stopped
    shaping height at all. See research_log/2026-09-07.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w.torch[:, 2]

    if target_height is None:
        target = env.reference_gait.sample_root_height()
        if target is None:
            target = torch.full_like(height, 1.05)
    else:
        target = torch.full_like(height, float(target_height))

    return torch.exp(-torch.square(height - target) / std**2)


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
    double_support_margin: float = 0.19,
    single_support_margin: float = -0.08,
    std: float = 0.05,
    foot_width: float = H1_FOOT_WIDTH_M,
    tipping_penalty_scale: float = 5.0,
    tipping_onset: float = 0.20,
) -> torch.Tensor:
    """Track the reference's mediolateral margin of stability *for the current support state*.

    Walking is not a single balance regime and this term used to treat it as one. Measured
    on the reference stride by MuJoCo FK, through the same convention
    :func:`compute_xcom_and_mos` uses:

    ========================  ==============  ==================
    regime                    share of cycle  reference MoS
    ========================  ==============  ==================
    double support            44.2%           **+0.191** +/- 0.024
    single support            55.8%           **-0.082** +/- 0.029
    ========================  ==============  ==================

    The two ranges do not overlap at all (+0.152..+0.233 against -0.141..-0.039), and
    contact state accounts for essentially all the variance: pooled within-regime standard
    deviation is 0.027 against 0.138 for the trajectory as a whole. One constant per regime
    therefore captures 96% of it.

    **What was wrong.** The old target was a single 0.04 m -- which is very close to the
    reference's *mean* (+0.0385), but the reference never sits at its mean; it alternates
    between two regimes the old term scored near zero. Worse, the formulation was one-sided:
    any margin at or above target scored a full 1.0, so over-stability was free and there
    was no gradient to come down, and a quadratic tipping penalty charged for *every*
    negative margin -- including the ones single support requires. Scored properly, the
    reference's own gait earned about 0.425 per stride against a shuffling policy's 0.601:
    **the term preferred the shuffle to the patient.** That is the shape of a reward that
    forbids single support, and it is the one candidate left after torque saturation,
    termination risk-aversion and clearance specification were each ruled out by measurement.

    Now: a two-sided Gaussian about the margin the reference actually held **at this gait
    phase**, read from the trajectory ``scripts/add_reference_mos.py`` stages into the stride
    archive, so neither over- nor under-stability is free.

    The two per-regime constants below are the fallback for archives that predate that
    trajectory. They capture the regime split but not the variation inside it -- the
    reference's within-regime spread is 0.024 and 0.029 m against this term's own 0.05 m
    width, so scored against constants the reference itself only reaches 0.788, with half the
    stride between 0.5 and 0.9. Against its own per-phase trajectory it reaches 1.0, which is
    what a target describing the demonstration should do.

    **The regime comes from the reference schedule, not from measured contact.** Measured
    gating was tried first, on the reasoning that this term should answer "are you
    appropriately stable for the stance you are in" and leave "are you in the right stance"
    to :func:`swing_timing`. It removed the penalty on single support but supplied no
    pressure toward it, and the policy simply stayed in double support: 84.7% of samples,
    double support unchanged at 0.847 against the reference's ~0.50.

    Schedule-gating is safe here even though the same choice broke
    :func:`paretic_foot_clearance`, and the difference is a measured cost asymmetry. There,
    the policy could satisfy a height target by rising onto its toe without unloading -- the
    fake was cheap. Here the fake would mean driving the XCoM 80 mm outside a support polygon
    that is 0.426 m wide with both feet down: measured over a 64-env rollout, **0.000%** of
    double-support samples reach -0.08 m, against **99.3%** of single-support samples. Genuine
    single support is the only affordable way to satisfy the single-support target.

    ``tipping_onset`` keeps a fall-arrest penalty, but only past -0.20 m, which is beyond
    anything the reference reaches (worst -0.141). Termination handles actual falls.

    Airborne samples score zero rather than being scored against either target: with no foot
    loaded there is no support state to be appropriate to, and the reference has no flight
    phase to imitate.
    """
    _, mos_lateral, in_contact = compute_xcom_and_mos(env, asset_cfg, sensor_cfg, foot_width=foot_width)

    num_loaded = in_contact.sum(dim=-1)

    target = env.reference_gait.sample_mos_target()
    if target is None:
        # No per-phase trajectory staged: fall back to one constant per scheduled regime.
        schedule = env.reference_gait.sample_contact()
        scheduled_double = schedule.sum(dim=-1) >= 1.5 if schedule is not None else num_loaded >= 2
        target = torch.where(
            scheduled_double,
            torch.full_like(mos_lateral, double_support_margin),
            torch.full_like(mos_lateral, single_support_margin),
        )

    tracking = torch.exp(-torch.square(mos_lateral - target) / std**2)
    tipping = torch.square(torch.clamp(-mos_lateral - tipping_onset, min=0.0))
    return (tracking - tipping_penalty_scale * tipping) * (num_loaded > 0).float()


def paretic_foot_clearance(
    env: H1PathologicalGaitEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.131,
    std: float = 0.04,
) -> torch.Tensor:
    """Reward clearance of the paretic foot through its *scheduled* swing.

    Foot drop is the deficit this task exists to reproduce, and the reference stride's
    joint angles alone do not tell the policy when the paretic foot is meant to be off
    the ground -- hence the phase gate.

    Two things here were measured wrong and are worth stating, because between them they
    made this term reward the deficit's opposite:

    **The target.** It was 0.10 m. Measured by MuJoCo FK on the reference stride, the
    paretic ankle link sits at 0.0706 m in stance and averages 0.1314 m through its
    scheduled swing (peak 0.1420) -- and Isaac's stance height for the same link is
    0.0704 m, so the two frames agree to 0.2 mm and the figures are directly comparable.
    At a 0.10 m target the reference's own swing scored 0.540 while a policy that lifted
    only 11 mm scored 0.807. The term preferred the under-lift. 0.131 m is the reference's
    own mean, so reproducing the patient is what maximises it.

    **The gate stays on measured contact, and that is deliberate.** It looks like a defect:
    a foot that never leaves the ground is never "in swing", so the term reads zero and
    offers no gradient to lift it. Switching it to the reference schedule was tried and
    regressed everything -- double support 0.824 -> 0.875, forward speed 0.163 -> 0.088 m/s,
    cost of transport 0.93 -> 3.19 -- while the paretic foot ended up *more* planted, loaded
    98.0% of scheduled swing against 94.8% before. Scoring height without requiring the foot
    to be unloaded pays the policy for raising the ankle link while the foot still bears
    load, which is expensive and is not a step. Requiring genuine unloading is what stops
    that; :func:`swing_timing` is the term that supplies the pressure to unload in the first
    place.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_cfg.name]

    foot_height = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids, 2]

    # body_ids are ordered (left, right); pick the paretic one per environment.
    is_right_paretic = (env.reference_gait.paretic_side > 0).long()
    index = is_right_paretic.unsqueeze(-1)
    paretic_height = torch.gather(foot_height, 1, index).squeeze(-1)

    net_force = torch.norm(sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids], dim=-1)
    paretic_force = torch.gather(net_force, 1, index).squeeze(-1)
    in_swing = (paretic_force <= 1.0).float()

    return torch.exp(-torch.square(paretic_height - target_height) / std**2) * in_swing


def paretic_load_aversion(
    env: H1PathologicalGaitEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Cost of committing body weight to the paretic limb while the sound limb is available.

    Give this a **negative** weight. Returns the paretic limb's share of total foot load in
    ``[0, 1]``, and **zero outside double support**.

    Why the term exists. Reduced paretic stance time is the defining temporal signature of
    hemiparetic gait: the reference loads the paretic limb 60.6% of the cycle against the
    sound limb's 83.5%, a stance-fraction asymmetry of -15.87%. Every policy this project has
    trained has the opposite sign -- AMP +2.02%, no-AMP +7.43%, no-AMP with full reference
    state initialisation +10.49%, three seeds each -- and the interventions that improved
    every other metric made this one steadily worse.

    The mechanism is that the impairment is modelled purely peripherally, as a 40% effort
    ceiling plus a stretch reflex. Under that model the paretic limb is the **cheaper limb to
    stand on**: standing costs almost nothing, and swinging is precisely what the weakened
    actuators cannot afford. Leaving it planted is optimal. Patients do the opposite, and the
    reason lives above the actuator level -- they will not commit weight to a limb they do not
    trust to carry them. Nothing in the environment represented that, so this term does.

    Why it is gated on double support. The weight-transfer *decision* only exists while both
    feet are down; during paretic single support there is no alternative, and charging for
    load there would penalise physics the policy cannot escape and reward falling toward the
    sound side. Gating also means the term cannot be satisfied by going airborne.

    Why load share rather than stance time. Charging for stance time directly would install
    the measured outcome by construction and prove nothing -- the question is whether the
    asymmetry *emerges* from an aversion to bearing weight, as the clinical account says it
    does. Load share and stance fraction are related but distinct, which leaves room for the
    informative failure: a policy that unloads the paretic foot while leaving it on the
    ground, satisfying the term without changing the gait. ``evaluate.py`` records
    ``paretic_load_share`` so that outcome is visible rather than mistaken for success.
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    force = torch.norm(sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids], dim=-1)

    # body_ids are ordered (left, right); swap for right-paretic environments so column 0 is
    # always the paretic limb. Averaging without this cancels the very asymmetry being shaped.
    is_right_paretic = (env.reference_gait.paretic_side > 0).unsqueeze(-1)
    force = torch.where(is_right_paretic, force.flip(-1), force)

    total = force.sum(dim=-1)
    both_loaded = (force > contact_threshold).all(dim=-1)
    share = force[:, 0] / total.clamp_min(1e-6)
    return torch.where(both_loaded, share, torch.zeros_like(share))


def spastic_torque_l2(env: H1PathologicalGaitEnv) -> torch.Tensor:
    """Squared TSRT reflex torque, as a diagnostic of how hard the policy fights spasticity."""
    return torch.sum(torch.square(env.applied_spastic_torque), dim=-1)
