# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hand-to-handle coupling and the Smart Walker's F/T sensor, as pure math.

Walker plan §2.3-2.4 (``docs/walker_evaluation_plan.md`` in the outer repo):

* **Grip coupling.** Each hand is tied to a point on the handlebar by a translational
  spring-damper, ``F = K (p_grip - p_hand) + C (v_grip - v_hand)``, applied every physics step:
  ``+F`` on the hand and ``-F`` on the handle. Two rigid 6-DoF welds to one bar would
  over-constrain the arms and give no readable force. Hands and wrists are not welds either.
* **The sensor-equivalent wrench.** The real walker has **one** ATI Axia on the middle of the
  bar. The wrench it reads is the net user force on the handle and its moment about the sensor
  origin, expressed in the frame ``force_torque_processor`` outputs and the admittance controller
  consumes: **x right, y up, z backward** (``walker_admittance.py``).
* **Per-hand reconstruction (W3).** Assuming two point grips with no grip moments, one 6-axis
  reading separates the hands' vertical loads (through the torque about the fore-aft axis) and
  their drive forces (through the yaw torque). It cannot separate their lateral forces. Simulation
  knows the per-hand forces exactly, so the reconstruction can be scored against them.

No Isaac imports: tested in ``tests/test_walker_coupling.py`` without booting the simulator.

**Walker body frame** (from the asset): +x is the direction of travel (the handle is at
x = -0.395 m, behind the base), +y is the walker's left, +z is up. So the processed frame is
x_p = -y_b, y_p = +z_b, z_p = -x_b: a proper rotation (determinant +1), so torques transform like
forces.
"""

from __future__ import annotations

import torch

#: Walker body frame -> the processor's output frame (x right, y up, z back). Rows are the
#: processed axes expressed in body coordinates.
BODY_TO_PROCESSED = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def grip_spring_forces(
    p_hand: torch.Tensor,
    v_hand: torch.Tensor,
    p_grip: torch.Tensor,
    v_grip: torch.Tensor,
    stiffness: float,
    damping: float,
) -> torch.Tensor:
    """Force the handle applies to each hand, ``(N, 2, 3)`` in world frame. Apply its negative to the handle.

    All inputs are ``(N, 2, 3)``, with hands ordered ``(left, right)`` and positions and
    velocities of the attachment points in world frame. The user's force on the handle is the
    negative of the return value, by Newton's third law, and it is what the sensor reads.
    """
    return stiffness * (p_grip - p_hand) + damping * (v_grip - v_hand)


def quat_rotate_inverse(quat_xyzw: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors ``(N, ..., 3)`` into the body frame of ``(N, 4)`` ``(x, y, z, w)`` quaternions."""
    q_vec = -quat_xyzw[:, :3]  # conjugate
    w = quat_xyzw[:, 3]
    while q_vec.dim() < vec.dim():
        q_vec = q_vec.unsqueeze(1)
        w = w.unsqueeze(1)
    t = 2.0 * torch.cross(q_vec.expand_as(vec), vec, dim=-1)
    return vec + w.unsqueeze(-1) * t + torch.cross(q_vec.expand_as(vec), t, dim=-1)


def sensor_wrench(
    user_force_on_handle_w: torch.Tensor,
    grip_pos_w: torch.Tensor,
    sensor_pos_w: torch.Tensor,
    walker_quat_w: torch.Tensor,
) -> torch.Tensor:
    """The wrench a mid-bar F/T sensor reads, ``(N, 6)`` = ``(Fx, Fy, Fz, Tx, Ty, Tz)``, processed frame.

    ``user_force_on_handle_w`` is ``(N, 2, 3)``, the force each hand applies to the bar.
    ``grip_pos_w`` is ``(N, 2, 3)``. ``sensor_pos_w`` is ``(N, 3)`` and ``walker_quat_w`` is ``(N, 4)``,
    ``(x, y, z, w)``. The moment is taken about the sensor origin. The handle's own inertia, 1 kg, is
    not included: it is what the physics-side joint-wrench cross-check is for (plan §2.4).
    """
    arm = grip_pos_w - sensor_pos_w.unsqueeze(1)
    force = user_force_on_handle_w.sum(dim=1)
    torque = torch.cross(arm, user_force_on_handle_w, dim=-1).sum(dim=1)
    rotation = BODY_TO_PROCESSED.to(force.device, force.dtype)
    force_p = quat_rotate_inverse(walker_quat_w, force) @ rotation.T
    torque_p = quat_rotate_inverse(walker_quat_w, torque) @ rotation.T
    return torch.cat((force_p, torque_p), dim=-1)


def reconstruct_hand_loads(
    wrench: torch.Tensor, half_grip_width: float, bar_offset_back: float, grip_up: float = 0.0
) -> dict[str, torch.Tensor]:
    """Per-hand vertical and drive forces from one sensor wrench (W3), processed frame.

    Assumes point grips at ``r = (+-d, u, h)`` from the sensor origin (right hand at +d), with no
    grip moments: ``d`` the half grip width, ``u`` the grips' height above the sensor axis
    (``grip_up``; the handle fitting puts it at 0.031 m) and ``h`` the bar's offset behind the
    sensor. With ``f_i = (fx, fy, fz)`` the user force at grip i, and ``F`` their sum:

        T_x = u F_z - h F_y                     -> redundant; returned as a consistency residual
        T_y = -d (fz_R - fz_L) + h F_x          -> drive split
        T_z =  d (fy_R - fy_L) - u F_x          -> vertical split

    Lateral forces are recoverable only as their sum. Returns the user's down and drive force per
    hand in the controller's sign convention (``f_down = -F_y``, ``f_drive = -F_z``).
    """
    fx, fy, fz, tx, ty, tz = wrench.unbind(-1)
    d, h, u = half_grip_width, bar_offset_back, grip_up
    fy_diff = (tz + u * fx) / d  # right - left
    fz_diff = (h * fx - ty) / d  # right - left
    fy_r, fy_l = (fy + fy_diff) / 2.0, (fy - fy_diff) / 2.0
    fz_r, fz_l = (fz + fz_diff) / 2.0, (fz - fz_diff) / 2.0
    return {
        "down_left": -fy_l,
        "down_right": -fy_r,
        "drive_left": -fz_l,
        "drive_right": -fz_r,
        "tx_residual": tx - u * fz + h * fy,
    }


def asymmetry_index(sound: torch.Tensor, paretic: torch.Tensor) -> torch.Tensor:
    """``(sound - paretic) / (sound + paretic)``: positive means the sound hand bears more. Plan §3.1."""
    return (sound - paretic) / (sound + paretic)
