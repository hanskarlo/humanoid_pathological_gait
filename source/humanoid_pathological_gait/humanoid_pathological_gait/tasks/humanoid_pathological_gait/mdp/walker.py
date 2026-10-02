# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Smart Walker in the loop: grip coupling, the real admittance law, and a velocity-held base.

A zero-dimensional action term, so it runs inside the action manager's per-physics-substep
``apply_actions`` -- the grip springs act at the physics rate (200 Hz) while the policy and the
admittance controller run at 50 Hz, as the real walker's controller does. The policy's action
space is unchanged.

Each physics substep:

1. hand attachment points (a fixed offset on each ``*_elbow_link``) and grip points (``+-d`` along
   the bar from the handle centre) in world frame, with their velocities;
2. grip spring-damper forces: ``+F`` applied to each hand at its attachment point, the net
   ``-F`` and its moment applied to the handle (``walker_coupling.grip_spring_forces``);
3. the walker base is held to the admittance command: planar velocity ``(v_x, omega_z)`` in the
   walker's heading, vertical velocity and roll/pitch rates left to physics (walker plan §2.2).

On every ``decimation``-th substep (50 Hz), the user's force on the handle becomes the mid-bar
sensor wrench (``walker_coupling.sensor_wrench``), passes through the processor and controller
ports (``walker_admittance``), and updates the command. The latest quantities are kept on the
environment as ``env.walker_state`` for the recorder.

**Unvalidated in simulation as written** (2026-10-01): written while the GPU was busy with H4.
Every constant marked MEASURE is a placeholder until the walker plan's validations V1-V7 pin it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .. import walker_admittance as wa
from .. import walker_coupling as wc

if TYPE_CHECKING:
    from ..h1_pathological_env import H1PathologicalGaitEnv


class WalkerCouplingAction(ActionTerm):
    """Couple the H1's hands to the walker handle and run the walker's admittance controller."""

    cfg: WalkerCouplingActionCfg

    def __init__(self, cfg: WalkerCouplingActionCfg, env: H1PathologicalGaitEnv):
        super().__init__(cfg, env)
        self._robot = env.scene[cfg.robot_name]
        self._walker = env.scene[cfg.asset_name]
        hand_ids, hand_names = self._robot.find_bodies(list(cfg.hand_body_names), preserve_order=True)
        if len(hand_ids) != 2:
            raise ValueError(f"expected two hand bodies {cfg.hand_body_names}, found {hand_names}")
        self._hand_ids = hand_ids
        self._handle_id = self._walker.find_bodies(cfg.handle_body_name)[0][0]
        self._sensor_id = self._walker.find_bodies(cfg.sensor_body_name)[0][0]
        n, dev = env.num_envs, env.device
        self._hand_offset = torch.tensor(cfg.hand_offset_body, device=dev).expand(n, 2, 3)
        # Grip points in the walker base frame, relative to the handle centre: left at +y, right at -y.
        d = cfg.half_grip_width
        h = cfg.grip_height_offset
        self._grip_offset_b = torch.tensor([[0.0, d, h], [0.0, -d, h]], device=dev).expand(n, 2, 3)
        self._processor = wa.ForceTorqueProcessor(n, dev, cfg.processor)
        self._controller = wa.WalkerAdmittance(n, dev, cfg.admittance)
        self._substep = 0
        self._decimation = env.cfg.decimation
        self._raw = torch.zeros(n, 0, device=dev)
        env.walker_state = {
            "hand_force_w": torch.zeros(n, 2, 3, device=dev),
            "spring_error": torch.zeros(n, 2, device=dev),
            "sensor_wrench": torch.zeros(n, 6, device=dev),
            "filtered_wrench": torch.zeros(n, 6, device=dev),
            "cmd": torch.zeros(n, 2, device=dev),
            "halt": torch.zeros(n, 3, dtype=torch.bool, device=dev),
        }

    # -- ActionTerm interface: no policy dimensions -------------------------------------------
    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._processor.reset(ids)
        self._controller.reset(ids)

    # -- the coupling -------------------------------------------------------------------------
    def _points(self):
        robot, walker = self._robot.data, self._walker.data
        hand_pos = robot.body_link_pos_w.torch[:, self._hand_ids]
        hand_quat = robot.body_link_quat_w.torch[:, self._hand_ids]
        hand_lin = robot.body_link_lin_vel_w.torch[:, self._hand_ids]
        hand_ang = robot.body_link_ang_vel_w.torch[:, self._hand_ids]
        n = hand_pos.shape[0]
        offset_w = _rotate(hand_quat.reshape(-1, 4), self._hand_offset.reshape(-1, 3)).reshape(n, 2, 3)
        p_hand = hand_pos + offset_w
        v_hand = hand_lin + torch.cross(hand_ang, offset_w, dim=-1)

        root_quat = walker.root_link_quat_w.torch
        handle_pos = walker.body_link_pos_w.torch[:, self._handle_id]
        grip_w = _rotate(root_quat.repeat_interleave(2, 0), self._grip_offset_b.reshape(-1, 3)).reshape(n, 2, 3)
        p_grip = handle_pos.unsqueeze(1) + grip_w
        handle_lin = walker.body_link_lin_vel_w.torch[:, self._handle_id].unsqueeze(1)
        handle_ang = walker.body_link_ang_vel_w.torch[:, self._handle_id].unsqueeze(1)
        v_grip = handle_lin + torch.cross(handle_ang.expand_as(grip_w), grip_w, dim=-1)
        return p_hand, v_hand, p_grip, v_grip, root_quat

    def apply_actions(self) -> None:
        p_hand, v_hand, p_grip, v_grip, root_quat = self._points()
        force_on_hand = wc.grip_spring_forces(p_hand, v_hand, p_grip, v_grip, self.cfg.stiffness, self.cfg.damping)
        user_on_handle = -force_on_hand

        self._robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force_on_hand,
            torques=torch.zeros_like(force_on_hand),
            positions=p_hand,
            body_ids=self._hand_ids,
            is_global=True,
        )
        handle_com = self._walker.data.body_com_pos_w.torch[:, self._handle_id]
        net = user_on_handle.sum(dim=1, keepdim=True)
        moment = torch.cross(p_grip - handle_com.unsqueeze(1), user_on_handle, dim=-1).sum(dim=1, keepdim=True)
        self._walker.permanent_wrench_composer.set_forces_and_torques_index(
            forces=net,
            torques=moment,
            body_ids=[self._handle_id],
            is_global=True,
        )

        state = self._env.walker_state
        state["hand_force_w"] = force_on_hand
        state["spring_error"] = torch.linalg.norm(p_grip - p_hand, dim=-1)

        if self._substep % self._decimation == 0:
            sensor_pos = self._walker.data.body_link_pos_w.torch[:, self._sensor_id]
            wrench = wc.sensor_wrench(user_on_handle, p_grip, sensor_pos, root_quat)
            filtered = self._processor.step(wrench)
            out = self._controller.step(filtered)
            state["sensor_wrench"] = wrench
            state["filtered_wrench"] = filtered
            state["cmd"] = torch.stack((out["v_x"], out["omega_z"]), dim=-1)
            state["halt"] = torch.stack((out["halt_deadman"], out["halt_collapse"], out["halt_impulse"]), dim=-1)
        self._substep += 1

        self._hold_base(root_quat, state["cmd"])

    def _hold_base(self, root_quat: torch.Tensor, cmd: torch.Tensor) -> None:
        """Planar velocity from the admittance command; vertical and roll/pitch left to physics."""
        x, y, z, w = root_quat.unbind(-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        lin = self._walker.data.root_lin_vel_w.torch.clone()
        ang = self._walker.data.root_ang_vel_w.torch.clone()
        lin[:, 0] = cmd[:, 0] * torch.cos(yaw)
        lin[:, 1] = cmd[:, 0] * torch.sin(yaw)
        ang[:, 2] = cmd[:, 1]
        self._walker.write_root_velocity_to_sim(torch.cat((lin, ang), dim=-1))


def reset_walker_ahead_of_robot(
    env: H1PathologicalGaitEnv,
    env_ids: torch.Tensor,
    handle_ahead: float = 0.35,
    handle_behind_base: float = 0.395,
    base_height: float = 0.322,
    asset_name: str = "walker",
) -> None:
    """Place the walker in front of the robot, facing its heading, at rest, wheels on the ground.

    Must run after ``reset_to_reference_pose``, whose written root pose it reads from
    ``env.reset_root_pose_w``. The handle centre goes ``handle_ahead`` metres in front of the
    pelvis along the robot's heading (MEASURE: the hands' forward reach at the reset pose). The
    walker base is ``handle_behind_base`` in front of its own handle (asset: handle at x = -0.395).
    ``base_height`` is the root link's (``base_link``) height above the ground with the wheels down:
    0.320 m measured at rest (handle 0.9325 m minus its 0.6125 m above ``base_link``) plus 2 mm of
    clearance. Not the default root pose: that is the asset's spawn origin, 0.93 m below
    ``base_link``, and writing it put the walker half through the ground (2026-10-02 smoke test).
    """
    walker = env.scene[asset_name]
    pose = env.reset_root_pose_w[env_ids]
    x, y, z, w = pose[:, 3:7].unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    heading = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    base_xy = pose[:, :2] + (handle_ahead + handle_behind_base) * heading
    root_z = env.scene.env_origins[env_ids, 2] + base_height
    half = 0.5 * yaw
    quat = torch.stack((torch.zeros_like(half), torch.zeros_like(half), torch.sin(half), torch.cos(half)), dim=-1)
    walker.write_root_pose_to_sim_index(
        root_pose=torch.cat((base_xy, root_z.unsqueeze(-1), quat), dim=-1), env_ids=env_ids
    )
    walker.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids
    )


def _rotate(quat_xyzw: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate body-frame vectors ``(M, 3)`` into the world by ``(M, 4)`` ``(x, y, z, w)`` quaternions."""
    q = quat_xyzw[:, :3]
    w = quat_xyzw[:, 3:4]
    t = 2.0 * torch.cross(q, vec, dim=-1)
    return vec + w * t + torch.cross(q, t, dim=-1)


@configclass
class WalkerCouplingActionCfg(ActionTermCfg):
    """Configuration for :class:`WalkerCouplingAction`."""

    class_type: type[ActionTerm] = WalkerCouplingAction
    asset_name: str = "walker"
    robot_name: str = "robot"
    hand_body_names: tuple[str, str] = ("left_elbow_link", "right_elbow_link")
    """Left first. The H1 has no hand link; the forearm's distal end is used (MEASURE the names)."""
    hand_offset_body: tuple[float, float, float] = (0.318, 0.0, -0.032)
    """The forearm tip in each elbow link's frame: twice the link's CoM offset, (0.159, 0, -0.016)
    measured in PhysX, for a 0.32 m forearm (scripts/measure_grip_posture.py, 2026-10-02)."""
    handle_body_name: str = "handle"
    sensor_body_name: str = "ft_sensor_link"
    half_grip_width: float = 0.179
    """Grip offset from the handle centre along the bar: the hands' half-separation at the grip
    posture (measured 0.179 m; the bar is 0.615 m long)."""
    grip_height_offset: float = 0.031
    """Grip points above the handle centre: the handle fitted to the H1's hands at the grip posture
    (0.964 m against the bar's 0.933 m), as walkers are fitted to the user's wrist height."""
    stiffness: float = 5000.0
    """N/m. Explicit stability at 5 ms needs omega*dt < 2; at 5 kN/m and ~1.5 kg, omega*dt ~ 0.29."""
    damping: float = 120.0
    """N s/m, ~0.7 critical for ~1.5 kg at 5 kN/m."""
    processor: wa.ForceTorqueProcessorParams = wa.ForceTorqueProcessorParams()
    admittance: wa.AdmittanceParams = wa.AdmittanceParams(enforce_halts=False)
    """Measurement mode by default: halts are recorded, not acted on (walker plan §2.5)."""
    debug_vis: bool = False
    clip: dict | None = None
