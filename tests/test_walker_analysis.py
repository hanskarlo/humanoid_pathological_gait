# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Walker metrics on hand-built rollouts: every number has a value it must come back as."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location("walker_analysis", SCRIPTS / "walker_analysis.py")
wa = importlib.util.module_from_spec(_spec)
sys.modules["walker_analysis"] = wa
_spec.loader.exec_module(wa)

DT = 0.02
GEOMETRY = {"half_grip_width": 0.179, "bar_offset_back": 0.0825, "grip_up": 0.031}


def rollout(steps=300, envs=2, down_lr=(30.0, 10.0), right_paretic=(False, True), yaw_rate=0.0, speed=0.2):
    """A walker pushed straight, with fixed per-hand down loads (left, right)."""
    hand = np.zeros((steps, envs, 2, 3))
    hand[..., 0, 2], hand[..., 1, 2] = down_lr
    wrench = np.zeros((steps, envs, 6))
    wrench[..., 1] = -(down_lr[0] + down_lr[1])  # F_y = -total down
    wrench[..., 5] = GEOMETRY["half_grip_width"] * (-down_lr[1] + down_lr[0])  # T_z = d (fy_R - fy_L)
    wrench[..., 2] = -12.0  # 12 N forward push
    cmd = np.zeros((steps, envs, 2))
    cmd[..., 0] = speed
    halt = np.zeros((steps, envs, 3), dtype=bool)
    t = np.arange(steps) * DT
    yaw = np.tile((yaw_rate * t)[:, None], (1, envs))
    xy = np.zeros((steps, envs, 2))
    xy[..., 0] = speed * t[:, None]
    valid = np.ones((steps, envs), dtype=bool)
    return dict(
        hand_force_b=hand, sensor_wrench=wrench, cmd=cmd, halt=halt, walker_xy=xy, walker_yaw=yaw, valid=valid,
        is_right_paretic=np.array(right_paretic), body_mass_kg=51.0, dt=DT, geometry=GEOMETRY,
    )  # fmt: skip


def test_asymmetry_is_paretic_signed_across_sides():
    """Left hand 30 N, right 10 N. Left-paretic env: paretic bears more (AI < 0); right-paretic: AI > 0."""
    left_paretic = wa.summarize_walker(**rollout(right_paretic=(False, False)))
    right_paretic = wa.summarize_walker(**rollout(right_paretic=(True, True)))
    assert left_paretic["ai_down"] == pytest.approx((10 - 30) / 40)
    assert right_paretic["ai_down"] == pytest.approx((30 - 10) / 40)


def test_load_drive_and_power():
    m = wa.summarize_walker(**rollout())
    assert m["handle_load_pct_bw_mean"] == pytest.approx(100 * 40 / (51.0 * 9.81))
    assert m["drive_effort_n"] == pytest.approx(12.0)
    assert m["drive_power_w"] == pytest.approx(12.0 * 0.2)


def test_the_settling_window_is_excluded():
    r = rollout()
    r["hand_force_b"][:100, ..., 2] = 999.0  # the first 2 s are a transient
    m = wa.summarize_walker(**r)
    assert m["handle_down_paretic_n"] < 100.0
    assert m["walker_valid_steps"] == pytest.approx((300 - 100) * 2)


def test_halts_are_counted_as_events_per_minute():
    r = rollout()
    r["halt"][150:160, 0, 0] = True  # one dead-man event, env 0
    r["halt"][200:210, 0, 0] = True  # a second
    m = wa.summarize_walker(**r)
    minutes = (300 - 100) * 2 * DT / 60.0
    assert m["halt_deadman_per_min"] == pytest.approx(2 / minutes)
    assert m["halt_collapse_per_min"] == 0.0


def test_veering_is_paretic_signed():
    """Turning left (CCW) at 0.2 rad/s: toward the paretic side for a left-paretic env, away for right."""
    left = wa.summarize_walker(**rollout(right_paretic=(False, False), yaw_rate=0.2))
    right = wa.summarize_walker(**rollout(right_paretic=(True, True), yaw_rate=0.2))
    assert left["veer_deg_per_m"] > 0 > right["veer_deg_per_m"]
    assert left["veer_deg_per_m"] == pytest.approx(-right["veer_deg_per_m"])


def test_single_sensor_reconstruction_recovers_the_split():
    m = wa.summarize_walker(**rollout())
    assert m["w3_down_rmse_n"] < 1e-9
    assert m["w3_ai_down"] == pytest.approx(m["ai_down"])


def test_paretic_stance_load_ratio():
    r = rollout(right_paretic=(False, False))
    contact = np.zeros((300, 2, 2), dtype=bool)
    contact[::2, :, 0] = True  # paretic single support on even steps
    contact[1::2, :, 1] = True  # sound single support on odd steps
    r["hand_force_b"][::2, ..., 2] *= 2.0  # twice the handle load during paretic stance
    m = wa.summarize_walker(**r, foot_contact=contact)
    assert m["paretic_stance_load_ratio"] == pytest.approx(2.0)
