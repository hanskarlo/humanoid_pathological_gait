# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL environment for 19-DoF H1 pathological (post-stroke) gait.

Everything the MDP terms need beyond stock Isaac Lab state lives here: the joint
ordering bridge to the clinical data, the reference stride playback, the TSRT
spasticity model, and the nominal actuator parameters that the asymmetric
randomization events perturb relative to.

Gait phase advances once per control step, before physics, so the reference the
action term drives toward is the pose the robot should reach by the end of the step
and the reward then scores how close it got.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .h1_joints import FOOT_BODY_NAMES, LEFT_LEG_JOINTS, RIGHT_LEG_JOINTS, H1JointLayout
from .reference import ReferenceGaitManager
from .tsrt import TSRTSpasticModel

if TYPE_CHECKING:
    from .config.h1_pathological.h1_pathological_env_cfg import H1PathologicalGaitEnvCfg


class H1PathologicalGaitEnv(ManagerBasedRLEnv):
    """H1 stroke-gait imitation environment with a spastic paretic limb."""

    cfg: H1PathologicalGaitEnvCfg

    def load_managers(self):
        """Set up gait state before the managers that read it are constructed.

        This hook runs after the scene exists and physics handles are live, but before
        the observation, reward, termination and curriculum managers are built -- and
        those managers evaluate their terms during construction, so the state they read
        has to exist by now.
        """
        robot = self.scene["robot"]

        self.joint_layout = H1JointLayout.from_sim_names(robot.joint_names, self.device)
        self.reference_gait = ReferenceGaitManager(
            stride_path=self.cfg.reference_stride_path,
            layout=self.joint_layout,
            num_envs=self.num_envs,
            device=self.device,
            stride_duration_s=self.cfg.reference_stride_duration_s,
        )
        self.tsrt_model = TSRTSpasticModel(self.joint_layout, self.cfg.tsrt_params, self.device)

        # Curriculum-controlled multiplier on the reflex torque.
        self.spasticity_scale = torch.full(
            (self.num_envs,), float(self.cfg.initial_spasticity_scale), device=self.device
        )
        self.applied_spastic_torque = torch.zeros((self.num_envs, robot.num_joints), device=self.device)

        # Nominal actuator/inertial parameters. The asymmetric randomization events scale
        # these rather than the live values, so repeated resets cannot compound drift.
        self.default_joint_stiffness = robot.data.default_joint_stiffness.torch.clone()
        self.default_joint_damping = robot.data.default_joint_damping.torch.clone()
        self.default_joint_effort_limits = robot.data.joint_effort_limits.torch.clone()
        self.default_body_mass = robot.data.body_mass.torch.clone()

        # Body indices used by the leg-mass randomization and the support-polygon terms.
        self.left_leg_body_ids = robot.find_bodies([f"{name}_link" for name in LEFT_LEG_JOINTS], preserve_order=True)[0]
        self.right_leg_body_ids = robot.find_bodies([f"{name}_link" for name in RIGHT_LEG_JOINTS], preserve_order=True)[
            0
        ]
        self.foot_body_ids = robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)[0]

        # Baseline push ranges, captured lazily by the push curriculum.
        self.base_push_velocity_range: dict[str, dict[str, tuple[float, float]]] = {}

        super().load_managers()

    def step(self, action: torch.Tensor):
        """Advance the gait clock, then step the environment."""
        self.reference_gait.advance(self.step_dt, self.cfg.gait_phase_rate_scale)
        return super().step(action)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset the given environments; gait phase and paretic side are reset by events."""
        super()._reset_idx(env_ids)
        self.applied_spastic_torque[env_ids] = 0.0

    """
    Accessors used by the external PPO+AMP training loop.
    """

    def get_current_reference_kinematics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference joint positions and velocities at the current gait phase."""
        return self.reference_gait.sample()

    def get_amp_kinematic_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The six kinematic components the AMP discriminator's feature extractor expects.

        Returns ``(root_height, projected_gravity, root_lin_vel_b, root_ang_vel_b,
        joint_pos, joint_vel)``, which concatenate to the 48-dimensional AMP feature.

        The joint tensors come back in **clinical** order, not simulation order. The
        expert motion corpus the discriminator scores against is stored clinically, and
        a discriminator handed two different joint permutations would separate agent from
        expert on the permutation alone, making the style reward meaningless.

        Root height is measured against the environment's own ground origin so it does
        not pick up the per-environment spawn offset.
        """
        robot = self.scene["robot"]
        root_height = robot.data.root_link_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        return (
            root_height,
            robot.data.projected_gravity_b.torch,
            robot.data.root_lin_vel_b.torch,
            robot.data.root_ang_vel_b.torch,
            self.joint_layout.to_clinical_order(robot.data.joint_pos.torch),
            self.joint_layout.to_clinical_order(robot.data.joint_vel.torch),
        )
