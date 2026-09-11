# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action term that drives the H1 as a residual on the clinical reference stride.

The policy does not command absolute joint angles. It commands a bounded residual
around the reference pose at the environment's current gait phase, which keeps the
search space near clinically plausible gait from the first iteration.

The same term injects the paretic limb's TSRT reflex torque as a feed-forward
effort. H1's joints use implicit (in-simulation PD) actuators, and Isaac Lab passes
``joint_efforts`` through to the solver alongside the position target, so the reflex
adds to the actuator torque rather than replacing it -- the policy has to work
against the spasticity, which is the point of the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from ..h1_pathological_env import H1PathologicalGaitEnv


class ReferenceResidualSpasticAction(JointPositionAction):
    """Reference-relative joint position targets plus TSRT reflex torque."""

    cfg: ReferenceResidualSpasticActionCfg
    _env: H1PathologicalGaitEnv

    def __init__(self, cfg: ReferenceResidualSpasticActionCfg, env: H1PathologicalGaitEnv):
        super().__init__(cfg, env)
        if self._num_joints != self._asset.num_joints:
            raise ValueError(
                "ReferenceResidualSpasticAction must cover every joint of the articulation so that its targets"
                f" line up with the reference stride; it resolved {self._num_joints} of {self._asset.num_joints}."
            )
        # Soft limits are constant per environment, so cache them once.
        self._soft_limits = self._asset.data.soft_joint_pos_limits.torch.clone()

        self._synergy_left = None
        self._synergy_right = None
        self._left_leg_columns = None
        self._right_leg_columns = None
        if cfg.paretic_synergy_rank is not None:
            self._build_synergy_projectors(cfg.paretic_synergy_rank)

    def _build_synergy_projectors(self, rank: int) -> None:
        """Cache the rank-limited projector for each paretic side.

        The reference manager returns the projector in the left-paretic frame. A right-paretic
        environment is the sagittal reflection, under which hip yaw and hip roll negate while
        the sagittal joints do not, so the projector is conjugated by that diagonal sign
        matrix: ``P_right = D P D``. ``P`` is symmetric and ``D`` is a diagonal of +-1, so the
        result is still a valid orthogonal projector of the same rank. Getting this wrong
        would impose a *mirrored* coordination constraint on half the environments -- the same
        class of error that cancelled 85% of the AMP corpus's asymmetry.
        """
        from ..reference import LEG_SYNERGY_JOINTS

        layout = self._env.joint_layout
        device = self._env.device
        projector = self._env.reference_gait.paretic_leg_synergy_projector(rank).to(device)

        self._left_leg_columns = torch.tensor(
            [layout.sim_names.index(f"left_{joint}") for joint in LEG_SYNERGY_JOINTS],
            dtype=torch.long,
            device=device,
        )
        self._right_leg_columns = torch.tensor(
            [layout.sim_names.index(f"right_{joint}") for joint in LEG_SYNERGY_JOINTS],
            dtype=torch.long,
            device=device,
        )
        signs = layout.mirror_sign[self._left_leg_columns].to(device)
        self._synergy_left = projector
        self._synergy_right = signs.unsqueeze(1) * projector * signs.unsqueeze(0)

    def process_actions(self, actions: torch.Tensor):
        """Turn raw actions into absolute joint targets around the reference pose."""
        self._raw_actions[:] = actions
        residual = self._raw_actions
        if self._synergy_left is not None:
            residual = self._apply_synergy_constraint(residual)
        q_ref, _ = self._env.reference_gait.sample()
        targets = q_ref + residual * self._scale
        self._processed_actions = torch.clamp(targets, self._soft_limits[..., 0], self._soft_limits[..., 1])

    def _apply_synergy_constraint(self, residual: torch.Tensor) -> torch.Tensor:
        """Restrict the paretic leg's residual to its rank-limited coordination subspace.

        Applied to the residual, not the target: the limb still follows the reference, it just
        cannot *correct* outside the pattern its own gait uses. The sound leg, the torso and
        both arms are untouched -- the deficit is unilateral.
        """
        constrained = residual.clone()
        is_right_paretic = (self._env.reference_gait.paretic_side > 0).unsqueeze(-1)

        left = residual[:, self._left_leg_columns] @ self._synergy_left
        right = residual[:, self._right_leg_columns] @ self._synergy_right
        constrained[:, self._left_leg_columns] = torch.where(
            is_right_paretic, residual[:, self._left_leg_columns], left
        )
        constrained[:, self._right_leg_columns] = torch.where(
            is_right_paretic, right, residual[:, self._right_leg_columns]
        )
        return constrained

    def apply_actions(self):
        """Write position targets, and the reflex torque when spasticity is enabled."""
        super().apply_actions()
        if not self.cfg.enable_spasticity:
            return
        tau_spastic = self._env.tsrt_model.compute(
            joint_pos=self._asset.data.joint_pos.torch,
            joint_vel=self._asset.data.joint_vel.torch,
            paretic_side=self._env.reference_gait.paretic_side,
            gain_scale=self._env.spasticity_scale,
        )
        self._asset.set_joint_effort_target_index(target=tau_spastic, joint_ids=self._joint_ids)
        self._env.applied_spastic_torque = tau_spastic


@configclass
class ReferenceResidualSpasticActionCfg(JointPositionActionCfg):
    """Configuration for :class:`ReferenceResidualSpasticAction`."""

    class_type: type[ActionTerm] = ReferenceResidualSpasticAction

    enable_spasticity: bool = True
    """Whether to apply the TSRT reflex torque on the paretic limb."""

    paretic_synergy_rank: int | None = None
    """Rank of the paretic leg's residual coordination subspace, or ``None`` for unconstrained.

    Models loss of selective motor control as a constraint on the *structure* of the paretic
    limb's corrections rather than on their magnitude, which is what the 40% effort ceiling
    already does. Clark et al. (J Neurophysiol, 2010) find post-stroke motor modules are merges
    of the healthy basis -- a rank reduction -- and that module count predicts walking
    performance. No prior work imposes this on an RL controller to reproduce post-stroke gait.

    5 is unconstrained. Measured on the best current policy, the fraction of paretic residual
    energy that survives the projection is 0.15 at rank 2, 0.28 at rank 3 and 0.36 at rank 4 --
    all *below* what a random subspace of the same rank would retain, because the policy's
    corrections are concentrated in hip yaw and hip roll while the reference's pattern is
    sagittal. So this is a strong constraint at any rank, and the rank must be chosen from a
    measured effect on this robot, not transferred from Clark's 3.6-versus-2.7 EMG modules.
    """

    use_default_offset: bool = False
    """Unused: the offset is the reference pose, resolved per step rather than at startup."""
