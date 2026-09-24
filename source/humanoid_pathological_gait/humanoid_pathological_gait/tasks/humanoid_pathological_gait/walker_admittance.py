# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Smart Walker's handle-force processing and admittance controller, batched over environments.

A port of the real walker's two ROS 2 nodes, so the simulated walker responds to a simulated push
the way the real one would:

* ``force_torque_processor`` (``smart-walker-ws/src/force_torque_processor/src/force_torque_processor.cpp``):
  tare offset, idle-bias tracking, per-axis deadband, and an exponential low-pass filter.
* ``sw_admittance_controller`` (``smart-walker-ws/src/sw_admittance_controller/src/admittance_controller.cpp``):
  force mapping, deadbands, hybrid steering, the three safety monitors, first-order admittance on
  forward speed and yaw rate, and the anti-spin kinematic clamp.

Pure torch. No Isaac imports, so it is tested against a line-by-line NumPy transcription of the
C++ (``tests/test_walker_admittance.py``) without booting the simulator. Both nodes run at
50 Hz on the robot, which is this task's control rate, so :meth:`WalkerAdmittance.step` is called
once per policy step.

**Frame.** Everything here is in the processor's output ("robot") frame, which is what the
controller subscribes to:

* ``y`` is **up**: pushing down on the handle makes ``F_y`` negative, hence ``f_down = -F_y``.
* ``z`` is **backward**, toward the user: pushing forward makes ``F_z`` negative, hence
  ``f_drive = -F_z``.
* ``x`` completes a right-handed frame, pointing to the walker's **right**. **Unverified against
  the hardware.** It enters only the lateral steering term, which carries weight
  ``1 - alpha = 0.1`` at the defaults.
* ``T_y`` is the torque about up, positive counter-clockwise seen from above, and it drives a
  positive (leftward) yaw rate.

The simulation layer builds this wrench from the per-hand coupling forces; see the walker plan,
§2.4.

**VAC obstacle terms are omitted.** There are no obstacles in the scene, so ``d_min`` never
enters ``d_safe`` and the real controller's effective damping equals its nominal damping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ForceTorqueProcessorParams:
    """``force_torque_processor`` defaults (``force_torque_processor.cpp`` L16-24)."""

    force_deadband_n: float = 1.0
    torque_deadband_nm: float = 0.05
    idle_force_threshold_n: float = 3.0
    idle_torque_threshold_nm: float = 0.15
    bias_alpha: float = 0.001
    enable_idle_bias_tracking: bool = True
    lpf_alpha: float = 0.2


@dataclass
class AdmittanceParams:
    """``sw_admittance_controller`` defaults (``admittance_controller.cpp`` L22-34)."""

    M_drive: float = 50.0
    B_drive: float = 60.0
    M_steer: float = 8.0
    B_steer: float = 7.0
    alpha: float = 0.9
    K_f: float = 1.0
    K_tau: float = 2.0
    R_min: float = 0.5
    deadman_min_force: float = 0.10
    collapse_max_force: float = 200.0
    impulse_threshold: float = 800.0
    dt: float = 0.02
    enforce_halts: bool = True
    """``True`` is the real controller: a halt zeroes the velocity state. ``False`` keeps the
    admittance running and only reports the halt. That is the walker plan's measurement mode,
    because a halt that stops the walker ends the rollout and hides how often halts fire."""


def _deadband(value: torch.Tensor, band: float) -> torch.Tensor:
    return torch.where(value.abs() < band, torch.zeros_like(value), value)


class ForceTorqueProcessor:
    """Batched ``force_torque_processor``, from the raw wrench in the robot frame to ``filtered_force``.

    The simulated sensor has no electrical offset, so the tare offset starts at zero, which is
    what a tare at rest measures in simulation. Idle-bias tracking is kept because it is real
    device behaviour: a light resting load under 3 N is slowly absorbed into the bias, which can
    walk ``f_down`` toward the dead-man threshold.
    """

    def __init__(self, num_envs: int, device: torch.device | str, params: ForceTorqueProcessorParams | None = None):
        self.params = params or ForceTorqueProcessorParams()
        self.offset = torch.zeros(num_envs, 6, device=device)
        self.filtered = torch.zeros(num_envs, 6, device=device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self.offset[env_ids] = 0.0
        self.filtered[env_ids] = 0.0

    def step(self, wrench: torch.Tensor) -> torch.Tensor:
        """Process one sample. ``wrench`` is ``(num_envs, 6)``, ``(Fx, Fy, Fz, Tx, Ty, Tz)``."""
        p = self.params
        calibrated = wrench - self.offset
        if p.enable_idle_bias_tracking:
            idle = (calibrated[:, :3].abs() < p.idle_force_threshold_n).all(dim=1) & (
                calibrated[:, 3:].abs() < p.idle_torque_threshold_nm
            ).all(dim=1)
            alpha = min(max(p.bias_alpha, 0.0), 1.0)
            updated = (1.0 - alpha) * self.offset + alpha * wrench
            self.offset = torch.where(idle.unsqueeze(1), updated, self.offset)
            calibrated = wrench - self.offset
        calibrated = torch.cat(
            (_deadband(calibrated[:, :3], p.force_deadband_n), _deadband(calibrated[:, 3:], p.torque_deadband_nm)),
            dim=1,
        )
        self.filtered = p.lpf_alpha * calibrated + (1.0 - p.lpf_alpha) * self.filtered
        return self.filtered


class WalkerAdmittance:
    """Batched ``sw_admittance_controller``: filtered handle wrench in, ``(v_x, omega_z)`` out."""

    def __init__(self, num_envs: int, device: torch.device | str, params: AdmittanceParams | None = None):
        self.params = params or AdmittanceParams()
        self.v_x = torch.zeros(num_envs, device=device)
        self.omega_z = torch.zeros(num_envs, device=device)
        self.prev_f_drive = torch.zeros(num_envs, device=device)
        self.prev_u_steer = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        for state in (self.v_x, self.omega_z, self.prev_f_drive, self.prev_u_steer):
            state[env_ids] = 0.0

    def step(self, filtered_wrench: torch.Tensor) -> dict[str, torch.Tensor]:
        """One 50 Hz control tick. Returns the commanded twist and the halt flags, per env.

        Follows ``control_loop`` (``admittance_controller.cpp`` L318-395) statement for statement,
        including two details that matter:

        * the derivative history is updated on every tick, halted or not;
        * the first tick after a reset differentiates against zero, so any push above
          ``impulse_threshold * dt`` (16 N) on the first tick trips the impulse monitor, as it does
          on the robot.
        """
        p = self.params
        f_drive = _deadband(-filtered_wrench[:, 2], 1.0)
        f_down = -filtered_wrench[:, 1]
        f_x = _deadband(filtered_wrench[:, 0], 1.0)
        tau_y = _deadband(filtered_wrench[:, 4], 0.05)

        u_steer = (1.0 - p.alpha) * (p.K_f * f_x) + p.alpha * (p.K_tau * tau_y)
        d_f_drive = (f_drive - self.prev_f_drive) / p.dt
        d_u_steer = (u_steer - self.prev_u_steer) / p.dt

        deadman = f_down < p.deadman_min_force
        collapse = f_down > p.collapse_max_force
        impulse = (d_f_drive.abs() > p.impulse_threshold) | (d_u_steer.abs() > p.impulse_threshold)
        halt = deadman | collapse | impulse

        v_next = self.v_x + ((f_drive - p.B_drive * self.v_x) / p.M_drive) * p.dt
        omega_next = self.omega_z + ((u_steer - p.B_steer * self.omega_z) / p.M_steer) * p.dt
        max_omega = v_next.abs() / p.R_min
        omega_next = torch.where(omega_next.abs() > max_omega, torch.copysign(max_omega, omega_next), omega_next)

        if p.enforce_halts:
            v_next = torch.where(halt, torch.zeros_like(v_next), v_next)
            omega_next = torch.where(halt, torch.zeros_like(omega_next), omega_next)

        self.v_x, self.omega_z = v_next, omega_next
        self.prev_f_drive, self.prev_u_steer = f_drive, u_steer
        return {
            "v_x": self.v_x,
            "omega_z": self.omega_z,
            "f_drive": f_drive,
            "f_down": f_down,
            "u_steer": u_steer,
            "halt_deadman": deadman,
            "halt_collapse": collapse,
            "halt_impulse": impulse,
        }
