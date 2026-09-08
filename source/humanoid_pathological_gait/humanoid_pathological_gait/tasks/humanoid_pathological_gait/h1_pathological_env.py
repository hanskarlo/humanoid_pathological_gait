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

        #: Fading balance assist. 1.0 means full help, 0.0 means the policy is on its own; the
        #: curriculum drives it down so the trained policy never depends on it.
        self.assist_scale = torch.tensor(float(self.cfg.initial_assist_scale), device=self.device)
        self._assist_force_magnitude = torch.zeros(self.num_envs, device=self.device)
        self._assist_body_ids = robot.find_bodies(["pelvis"], preserve_order=True)[0]
        self._assist_foot_ids = robot.find_bodies([".*ankle_link"], preserve_order=True)[0]

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

    def _apply_balance_assist(self) -> None:
        """Apply a fading mediolateral assist wrench at the pelvis.

        The policy can enter single support but cannot hold it: measured across every run it
        reaches a lateral margin of -0.23 m there against the patient's -0.082, i.e. it falls
        sideways and plants the foot to recover. High double support is that recovery. Nine
        interventions that changed what behaviour *costs* -- eight reward terms and one
        actuator retune -- failed to move it, because reward shaping cannot install a skill
        the policy never successfully executes: every attempt at single support ends in a
        near-fall, so there is no gradient toward a good one.

        This changes what is *reachable* instead. A lateral restoring force at the pelvis
        holds the robot up through the unstable phase so the policy can experience controlled
        single support and learn the control, then decays to zero so the final policy stands
        on its own. Precedent: Shi et al. (ISRR 2022), and A2CF (Cao et al., arXiv 2506.23125)
        with a decaying 6D pelvis wrench on a 29-DoF humanoid.

        The force opposes the *lateral* CoM velocity and offset only. It deliberately does not
        assist forward progress or vertical support: those are not the missing skill, and
        holding the robot up would let it collect the alive bonus for free.
        """
        scale = float(self.assist_scale)
        if scale <= 0.0:
            return

        robot = self.scene["robot"]
        # Work in the robot's yaw frame so "lateral" tracks the walking direction.
        quat = robot.data.root_link_quat_w.torch
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cos, sin = torch.cos(yaw), torch.sin(yaw)

        mass = robot.data.body_mass.torch.unsqueeze(-1)
        com_vel_w = (robot.data.body_com_lin_vel_w.torch * mass).sum(dim=1) / mass.sum(dim=1)
        com_pos_w = (robot.data.body_com_pos_w.torch * mass).sum(dim=1) / mass.sum(dim=1)
        foot_pos_w = robot.data.body_link_pos_w.torch[:, self._assist_foot_ids]

        # Where the CoM *should* be laterally: over the feet the reference schedule says are
        # down. Targeting the midpoint of both feet instead would pull the robot back toward
        # double support -- assisting the very behaviour this is meant to break it out of, and
        # measurably so: the first version shortened episodes instead of lengthening them.
        schedule = self.reference_gait.sample_contact()
        if schedule is not None:
            # sample_contact returns (paretic, sound); foot_pos_w is (left, right).
            is_right_paretic = (self.reference_gait.paretic_side > 0).unsqueeze(-1)
            weights = torch.where(is_right_paretic, schedule.flip(-1), schedule)
            total = weights.sum(dim=-1, keepdim=True).clamp(min=1e-3)
            target = (foot_pos_w * weights.unsqueeze(-1)).sum(dim=1) / total
        else:
            target = foot_pos_w.mean(dim=1)
        delta = com_pos_w[:, :2] - target[:, :2]
        offset_y = -sin * delta[:, 0] + cos * delta[:, 1]
        vel_y = -sin * com_vel_w[:, 0] + cos * com_vel_w[:, 1]

        magnitude = -(self.cfg.assist_stiffness * offset_y + self.cfg.assist_damping * vel_y)
        magnitude = magnitude.clamp(-self.cfg.assist_max_force, self.cfg.assist_max_force) * scale

        forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        forces[:, 0, 0] = -sin * magnitude
        forces[:, 0, 1] = cos * magnitude
        self._assist_force_magnitude = magnitude

        composer = robot.permanent_wrench_composer
        composer.reset()
        composer.add_forces_and_torques_index(
            forces=forces, torques=torch.zeros_like(forces), body_ids=self._assist_body_ids
        )

    def step(self, action: torch.Tensor):
        """Advance the gait clock, then step the environment."""
        self.reference_gait.advance(self.step_dt, self.cfg.gait_phase_rate_scale)
        self._apply_balance_assist()
        # The reference's root displacement restarts at zero each stride, so the robot's
        # origin has to restart with it -- otherwise the tracking error grows without bound
        # across cycles instead of measuring progress within one.
        wrapped = self.reference_gait.cycle_wrapped
        if wrapped.any():
            self._anchor_root_cycle(wrapped.nonzero(as_tuple=False).squeeze(-1))
        return super().step(action)

    def _anchor_root_cycle(self, env_ids) -> None:
        """Pin the root-displacement origin to where these environments are now."""
        robot = self.scene["robot"]
        position = robot.data.root_link_pos_w.torch[env_ids, :2] - self.scene.env_origins[env_ids, :2]
        quat = robot.data.root_link_quat_w.torch[env_ids]
        # Isaac Lab quaternions are (x, y, z, w) in this release.
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        self.reference_gait.set_cycle_anchor(position, yaw, env_ids)

    def root_progression_error(self) -> torch.Tensor | None:
        """``(N, 2)`` along-track and cross-track error against the reference root."""
        robot = self.scene["robot"]
        position = robot.data.root_link_pos_w.torch[:, :2] - self.scene.env_origins[:, :2]
        quat = robot.data.root_link_quat_w.torch
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return self.reference_gait.root_progression_error(position, yaw)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset the given environments; gait phase and paretic side are reset by events."""
        super()._reset_idx(env_ids)
        self.applied_spastic_torque[env_ids] = 0.0
        # Events have already placed the robot and drawn a new start phase, so anchor here
        # rather than in the event: the reference displacement is measured from phase 0 and
        # a randomised start phase would otherwise be scored against the wrong origin.
        self._anchor_root_cycle(env_ids)

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
