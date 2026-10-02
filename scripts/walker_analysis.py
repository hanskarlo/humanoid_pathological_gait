# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Walker metrics on a recorded coupled rollout: pure NumPy, no simulator.

The reductions behind every walker number in the paper, defined in
``docs/walker_evaluation_plan.md`` §3 (outer repo) before any coupled patient rollout was read.
Kept apart from the simulator for the same reason as ``gait_analysis.py``: the math can be checked
against hand-built rollouts (``tests/test_walker_analysis.py``).

Conventions:

* hands are recorded ``(left, right)``. They are reordered to ``(paretic, sound)`` with
  ``is_right_paretic``, so left- and right-paretic environments pool;
* ``hand_force_b`` is the force the handle applies **to the hand**, in the **walker's body frame**
  (+z up along the walker). The user's downward load on the handle is therefore its **+z**
  component (the user pushes the handle down; the handle pushes the hand up). The walker frame,
  not the world, because that is the load the device carries and the axis its sensor measures:
  with the walker tipped the two differ (2026-10-02);
* ``sensor_wrench`` is in the processor frame the real controller reads (x right, y up, z back), so
  ``f_drive = -F_z`` and the yaw torque is ``T_y`` (counter-clockwise positive);
* "paretic-signed" quantities flip sign on right-paretic environments, so positive always means
  toward the same side relative to the impairment: **positive = toward the paretic side**;
* only the steady-state window counts: valid steps from ``settle_s`` (default 2 s) on. That
  excludes the coupling's 0.2 s reset transient, and then some.
"""

from __future__ import annotations

import numpy as np

GRAVITY = 9.81


def paretic_first(values: np.ndarray, is_right_paretic: np.ndarray, axis: int = 2) -> np.ndarray:
    """Reorder a ``(left, right)`` axis to ``(paretic, sound)`` per environment."""
    flipped = np.flip(values, axis=axis)
    shape = [1] * values.ndim
    shape[1] = values.shape[1]
    return np.where(np.asarray(is_right_paretic).reshape(shape), flipped, values)


def reconstruct_down_forces(wrench: np.ndarray, half_grip_width: float, bar_offset_back: float, grip_up: float):
    """Per-hand down force ``(left, right)`` from the sensor wrench alone (W3). Mirrors
    ``walker_coupling.reconstruct_hand_loads``: ``fy_R - fy_L = (T_z + u F_x) / d``."""
    fx, fy, tz = wrench[..., 0], wrench[..., 1], wrench[..., 5]
    diff = (tz + grip_up * fx) / half_grip_width  # fy_right - fy_left
    fy_r, fy_l = (fy + diff) / 2.0, (fy - diff) / 2.0
    return np.stack((-fy_l, -fy_r), axis=-1)


def summarize_walker(
    hand_force_b: np.ndarray,
    sensor_wrench: np.ndarray,
    cmd: np.ndarray,
    halt: np.ndarray,
    walker_xy: np.ndarray,
    walker_yaw: np.ndarray,
    valid: np.ndarray,
    is_right_paretic: np.ndarray,
    body_mass_kg: float,
    dt: float,
    geometry: dict[str, float],
    foot_contact: np.ndarray | None = None,
    settle_s: float = 2.0,
) -> dict[str, float]:
    """The walker plan's §3 metrics. Shapes: ``(T, N, ...)`` with T steps and N environments.

    ``foot_contact`` is ``(T, N, 2)`` ``(left, right)``, optional; with it the paretic-stance load
    ratio is reported. ``geometry`` holds ``half_grip_width``, ``bar_offset_back`` and ``grip_up``
    for the W3 reconstruction.
    """
    steps = hand_force_b.shape[0]
    window = valid.copy()
    window[: min(steps, int(round(settle_s / dt)))] = False
    if not window.any():
        return {"walker_valid_steps": 0.0}

    down = paretic_first(hand_force_b[..., 2], is_right_paretic, axis=2)  # (T, N, 2) paretic, sound
    down_p, down_s = down[..., 0][window], down[..., 1][window]
    total = down_p + down_s
    weight = body_mass_kg * GRAVITY
    side = np.where(np.asarray(is_right_paretic), -1.0, 1.0)[None, :]

    out = {
        "walker_valid_steps": float(window.sum()),
        "handle_down_paretic_n": float(down_p.mean()),
        "handle_down_sound_n": float(down_s.mean()),
        # Primary (W1): positive = the sound hand bears more.
        "ai_down": float((down_s.mean() - down_p.mean()) / (down_s.mean() + down_p.mean())),
        "handle_load_pct_bw_mean": float(100.0 * total.mean() / weight),
        "handle_load_pct_bw_p95": float(100.0 * np.percentile(total, 95) / weight),
        "drive_effort_n": float((-sensor_wrench[..., 2])[window].mean()),
        "drive_power_w": float((-sensor_wrench[..., 2] * cmd[..., 0])[window].mean()),
        # T_y counter-clockwise positive; on a left-paretic env, CCW turns toward the paretic (left)
        # side, so paretic-signed = T_y for left-paretic, -T_y for right-paretic.
        "steering_bias_nm": float((sensor_wrench[..., 4] * side)[window].mean()),
    }

    # Halts per minute of valid steady-state time, counted as rising edges.
    for k, name in enumerate(("deadman", "collapse", "impulse")):
        flag = halt[..., k] & window
        rising = flag[1:] & ~flag[:-1]
        out[f"halt_{name}_per_min"] = float(rising.sum() / max(window.sum() * dt / 60.0, 1e-9))

    # Veering, per environment over its steady window: heading change per metre travelled, and
    # final lateral offset from the starting heading line. Both paretic-signed.
    drift, lateral = [], []
    for env in range(walker_xy.shape[1]):
        idx = np.flatnonzero(window[:, env])
        if idx.size < 2:
            continue
        xy = walker_xy[idx, env]
        yaw = np.unwrap(walker_yaw[idx, env])
        path = np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=-1))
        start_heading = np.array([np.cos(yaw[0]), np.sin(yaw[0])])
        offset = xy[-1] - xy[0]
        cross = start_heading[0] * offset[1] - start_heading[1] * offset[0]  # + = to the left
        s = side[0, env]
        if path > 0.05:
            drift.append(s * np.degrees(yaw[-1] - yaw[0]) / path)
        lateral.append(s * cross)
    out["veer_deg_per_m"] = float(np.mean(drift)) if drift else float("nan")
    out["lateral_deviation_m"] = float(np.mean(lateral)) if lateral else float("nan")

    # W3: the single mid-bar sensor's per-hand split against the ground truth.
    recon = paretic_first(
        reconstruct_down_forces(
            sensor_wrench, geometry["half_grip_width"], geometry["bar_offset_back"], geometry["grip_up"]
        ),
        is_right_paretic,
        axis=2,
    )
    rp, rs = recon[..., 0][window], recon[..., 1][window]
    out["w3_down_rmse_n"] = float(np.sqrt(np.mean(np.square(np.stack((rp - down_p, rs - down_s))))))
    out["w3_ai_down"] = float((rs.mean() - rp.mean()) / (rs.mean() + rp.mean()))

    if foot_contact is not None:
        contact = paretic_first(foot_contact.astype(bool), is_right_paretic, axis=2)
        single_p = contact[..., 0] & ~contact[..., 1] & window
        single_s = contact[..., 1] & ~contact[..., 0] & window
        tot = down[..., 0] + down[..., 1]
        if single_p.any() and single_s.any():
            out["paretic_stance_load_ratio"] = float(tot[single_p].mean() / tot[single_s].mean())
        else:
            out["paretic_stance_load_ratio"] = float("nan")
    return out
