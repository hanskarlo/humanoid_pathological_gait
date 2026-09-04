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

    def process_actions(self, actions: torch.Tensor):
        """Turn raw actions into absolute joint targets around the reference pose."""
        self._raw_actions[:] = actions
        q_ref, _ = self._env.reference_gait.sample()
        targets = q_ref + self._raw_actions * self._scale
        self._processed_actions = torch.clamp(targets, self._soft_limits[..., 0], self._soft_limits[..., 1])

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

    use_default_offset: bool = False
    """Unused: the offset is the reference pose, resolved per step rather than at startup."""
