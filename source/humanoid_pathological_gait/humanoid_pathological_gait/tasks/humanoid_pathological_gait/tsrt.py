# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Tonic Stretch Reflex Threshold (TSRT) spasticity model for the paretic limb.

References: Feldman (1986), Levin & Feldman (1994), Musampa et al. (2007).

A spastic muscle recruits when the joint is stretched past a threshold angle, and
that threshold falls as the stretch velocity rises -- which is why spasticity is
velocity-dependent. For a joint whose stretch coordinate is :math:`\theta` with
stretch velocity :math:`\dot{\theta}^{+} = \max(\dot{\theta}, 0)`:

.. math::

    \theta_{th} &= \lambda_0 - \mu \dot{\theta}^{+} \\
    \tau &= -\left(k \max(\theta - \theta_{th},\, 0) + b\, \dot{\theta}^{+}\right)

The torque opposes the stretch, and only engages once the joint is both past the
threshold and actively lengthening. Three joints per paretic leg are modelled:

* **ankle** -- plantarflexor spasticity resisting dorsiflexion (foot drop),
* **knee** -- resists flexion (stiff-knee gait),
* **hip roll** -- adductor spasticity resisting abduction (scissoring/circumduction).

Each is expressed through a *stretch sign*: the sign that converts the joint's own
coordinate into the stretch coordinate. Mirroring a joint to the other leg flips
that sign for the roll axes and leaves the pitch axes alone, exactly as in
:mod:`~.h1_joints`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .h1_joints import H1JointLayout


@dataclass
class TSRTParams:
    """Per-joint-group biomechanical parameters of the TSRT model.

    ``lambda_0`` is the resting threshold angle (rad), ``mu`` the velocity
    sensitivity of that threshold (rad per rad/s), ``k`` the reflex stiffness
    (Nm/rad) and ``b`` the reflex damping (Nm per rad/s).
    """

    lambda_0_ankle: float = -0.10

    #: Knee reflex threshold, rad. 0.25 rad is 14.3 deg, which sits just under the paretic
    #: knee's own peak flexion in the reference stride (14.80 deg).
    #:
    #: It was 0.35 rad (20.05 deg) and that is *above* anything the reference does, so the
    #: reflex could never shape the stride -- it only engaged once the policy had already
    #: flexed past reference motion, as a late brake on gross deviation. Measured: the
    #: reference crosses the old threshold on 0.0% of the stride, and a trained policy
    #: reached 40 deg of paretic knee flexion against the reference's 14.8 deg, losing the
    #: stiff-knee pathology entirely (symmetry index 40.7% against the reference's 93.2%).
    #:
    #: Lowering the threshold rather than raising ``k`` is the clinically canonical way to
    #: express severity here: in Levin & Feldman's framework spasticity *is* a reduced
    #: tonic stretch reflex threshold, recruited within the functional range of motion
    #: instead of only at the extremes. At 0.25 rad the reference itself crosses it on 3.8%
    #: of the stride, so reference motion stays achievable while excess flexion is resisted
    #: from the moment it exceeds what the patient actually did.
    lambda_0_knee: float = 0.25

    #: Hip-roll reflex threshold, rad. -0.0435 rad is -2.49 deg, just under the reference
    #: paretic hip's peak abduction of -2.42 deg, and crossed on 4.4% of the stride -- the same
    #: placement rule applied to the knee, which the reference crosses on 3.8%.
    #:
    #: It was 0.20 rad (+11.46 deg). The reference paretic hip roll spans -7.89..-2.42 deg, so
    #: it stays *adducted* throughout and never came within 14 deg of that threshold. The
    #: reflex was therefore unable to shape reference-tracking motion -- but it was **not**
    #: inert, and an earlier version of this comment wrongly said so: measured on a trained
    #: policy it engaged on 24.3% of samples, because the policy's own hip deviates far beyond
    #: anything the reference does. It was acting as a brake on excess abduction rather than a
    #: shaper of gait, exactly as the knee reflex was before its threshold was corrected.
    #:
    #: At the new threshold the engagement rate is similar (25.6%) but the torque roughly
    #: doubles: mean 4.19 -> 9.22 Nm, p99 24.9 -> 44.1 Nm.
    #:
    #: A caution that belongs with this parameter. The reference achieves its lateral foot
    #: clearance by **pelvic hiking, not hip abduction** -- coronal pelvic obliquity rises from
    #: +4.25 deg in paretic stance to +8.75 in paretic swing, a +4.50 deg hiking signature,
    #: while the hip's entire range of motion is 5.47 deg and never leaves adduction. So this
    #: reflex firing is necessary for adductor spasticity to be modelled at all, but it is
    #: *not* the mechanism behind circumduction in this data, and a claim that circumduction
    #: emerges from hip adductor spasticity is not supported by the reference kinematics.
    lambda_0_hip: float = -0.0435

    mu_ankle: float = 0.08
    mu_knee: float = 0.06
    mu_hip: float = 0.05

    k_spastic_ankle: float = 85.0
    b_spastic_ankle: float = 6.5
    k_spastic_knee: float = 120.0
    b_spastic_knee: float = 8.0
    k_spastic_hip: float = 90.0
    b_spastic_hip: float = 6.0

    velocity_deadband: float = 0.05
    """Stretch velocity (rad/s) below which the reflex stays silent."""


@dataclass
class _SpasticJoint:
    """One modelled joint: where it sits in the articulation and how it stretches."""

    sim_id: int
    stretch_sign: float
    lambda_0: float
    mu: float
    stiffness: float
    damping: float


class TSRTSpasticModel:
    """Vectorised TSRT reflex torques for a batch of environments.

    The model is built once against a :class:`~.h1_joints.H1JointLayout` and then
    evaluated every physics step. Torques are returned in simulation joint order
    with zeros on every non-spastic joint, ready to hand to
    ``Articulation.set_joint_effort_target_index``.
    """

    def __init__(self, layout: H1JointLayout, params: TSRTParams, device: torch.device | str):
        self.layout = layout
        self.params = params
        self.device = torch.device(device)

        # Stretch signs are written for the LEFT leg; the right leg inherits them through
        # the layout's mirror signs, so the two sides stay consistent by construction.
        left_specs = {
            "ankle": (-1.0, params.lambda_0_ankle, params.mu_ankle, params.k_spastic_ankle, params.b_spastic_ankle),
            "knee": (1.0, params.lambda_0_knee, params.mu_knee, params.k_spastic_knee, params.b_spastic_knee),
            "hip_roll": (1.0, params.lambda_0_hip, params.mu_hip, params.k_spastic_hip, params.b_spastic_hip),
        }

        self._left_joints: list[_SpasticJoint] = []
        self._right_joints: list[_SpasticJoint] = []
        for suffix, (sign, lam, mu, k, b) in left_specs.items():
            left_id = layout.index_of(f"left_{suffix}")
            right_id = int(layout.mirror_index[left_id])
            mirror_sign = float(layout.mirror_sign[left_id])
            self._left_joints.append(_SpasticJoint(left_id, sign, lam, mu, k, b))
            self._right_joints.append(_SpasticJoint(right_id, sign * mirror_sign, lam, mu, k, b))

    def compute(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        paretic_side: torch.Tensor,
        gain_scale: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """Compute reflex torques for every environment.

        Args:
            joint_pos: Joint positions in simulation order. Shape ``(num_envs, 19)``.
            joint_vel: Joint velocities in simulation order. Shape ``(num_envs, 19)``.
            paretic_side: ``-1`` for a left-paretic environment, ``+1`` for right-paretic.
                Shape ``(num_envs,)``.
            gain_scale: Multiplier on the reflex torque, used by the spasticity
                curriculum to fade the deficit in. Scalar or shape ``(num_envs,)``.

        Returns:
            Reflex torques in simulation order. Shape ``(num_envs, 19)``.
        """
        tau = torch.zeros_like(joint_pos)

        # Only the paretic leg is spastic; the sound leg contributes nothing.
        is_left_paretic = (paretic_side < 0).float()
        for side_joints, side_active in (
            (self._left_joints, is_left_paretic),
            (self._right_joints, 1.0 - is_left_paretic),
        ):
            for joint in side_joints:
                pos = joint.stretch_sign * joint_pos[:, joint.sim_id]
                vel = joint.stretch_sign * joint_vel[:, joint.sim_id]
                stretch_vel = torch.clamp(vel, min=0.0)
                threshold = joint.lambda_0 - joint.mu * stretch_vel
                overstretch = torch.clamp(pos - threshold, min=0.0)
                engaged = (overstretch > 0.0) & (stretch_vel > self.params.velocity_deadband)
                # Opposes the stretch, so negative in the stretch coordinate; the sign
                # flip carries it back into the joint's own coordinate.
                reflex = -(joint.stiffness * overstretch + joint.damping * stretch_vel) * engaged.float()
                tau[:, joint.sim_id] = joint.stretch_sign * reflex * side_active

        if isinstance(gain_scale, torch.Tensor):
            gain_scale = gain_scale.unsqueeze(-1)
        return tau * gain_scale
