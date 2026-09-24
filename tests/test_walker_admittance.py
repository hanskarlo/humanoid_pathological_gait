# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity of the batched walker admittance port against a transcription of the real C++ nodes.

The reference below is a scalar, statement-for-statement transcription of
``force_torque_processor.cpp`` (``topic_callback``, L279-375) and
``admittance_controller.cpp`` (``control_loop``, L318-395), VAC terms omitted as in the port.
The port must agree with it to 1e-9 in float64 on every scripted case, and the physics it
produces must match what the real controller's equations imply (walker plan, validations V4, V5
and V7).
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

MODULE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "source/humanoid_pathological_gait/humanoid_pathological_gait/tasks/humanoid_pathological_gait"
    / "walker_admittance.py"
)
# Loaded by path: the module is pure torch, and importing it through the package would pull in Isaac Lab.
_spec = importlib.util.spec_from_file_location("walker_admittance", MODULE)
wa = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = wa  # dataclasses resolve postponed annotations through sys.modules
_spec.loader.exec_module(wa)


class ReferenceProcessor:
    """``force_torque_processor.cpp`` after calibration, for one sensor."""

    def __init__(self, p):
        self.p = p
        self.offset = [0.0] * 6
        self.filtered = [0.0] * 6

    @staticmethod
    def deadband(value, band):
        return 0.0 if abs(value) < band else value

    def step(self, t):
        p = self.p
        cal = [t[i] - self.offset[i] for i in range(6)]
        idle = all(abs(cal[i]) < p.idle_force_threshold_n for i in range(3)) and all(
            abs(cal[i]) < p.idle_torque_threshold_nm for i in range(3, 6)
        )
        if p.enable_idle_bias_tracking and idle:
            a = min(max(p.bias_alpha, 0.0), 1.0)
            self.offset = [(1.0 - a) * self.offset[i] + a * t[i] for i in range(6)]
            cal = [t[i] - self.offset[i] for i in range(6)]
        cal = [self.deadband(cal[i], p.force_deadband_n) for i in range(3)] + [
            self.deadband(cal[i], p.torque_deadband_nm) for i in range(3, 6)
        ]
        self.filtered = [p.lpf_alpha * cal[i] + (1.0 - p.lpf_alpha) * self.filtered[i] for i in range(6)]
        return list(self.filtered)


class ReferenceController:
    """``admittance_controller.cpp`` ``control_loop``, VAC inactive."""

    def __init__(self, p):
        self.p = p
        self.v_x = 0.0
        self.omega_z = 0.0
        self.prev_f_drive = 0.0
        self.prev_u_steer = 0.0

    @staticmethod
    def deadband(value, band):
        return 0.0 if abs(value) < band else value

    def step(self, w):
        p = self.p
        f_drive = -w[2]
        f_down = -w[1]
        f_x = w[0]
        tau_y = w[4]
        f_drive = self.deadband(f_drive, 1.0)
        f_x = self.deadband(f_x, 1.0)
        tau_y = self.deadband(tau_y, 0.05)
        u_steer = ((1.0 - p.alpha) * (p.K_f * f_x)) + (p.alpha * (p.K_tau * tau_y))
        d_f = (f_drive - self.prev_f_drive) / p.dt
        d_u = (u_steer - self.prev_u_steer) / p.dt
        halt = False
        if f_down < p.deadman_min_force or f_down > p.collapse_max_force:
            halt = True
        if abs(d_f) > p.impulse_threshold or abs(d_u) > p.impulse_threshold:
            halt = True
        if halt:
            self.v_x = 0.0
            self.omega_z = 0.0
        else:
            self.v_x = self.v_x + ((f_drive - (p.B_drive * self.v_x)) / p.M_drive) * p.dt
            self.omega_z = self.omega_z + ((u_steer - (p.B_steer * self.omega_z)) / p.M_steer) * p.dt
            max_omega = abs(self.v_x) / p.R_min
            if abs(self.omega_z) > max_omega:
                self.omega_z = math.copysign(max_omega, self.omega_z)
        self.prev_f_drive = f_drive
        self.prev_u_steer = u_steer
        return self.v_x, self.omega_z, halt


def wrench(push=0.0, down=0.0, lateral=0.0, yaw=0.0):
    """A handle wrench in the processor's frame from user-intuitive quantities.

    Forward push is -F_z, downward load is -F_y, yaw torque is +T_y (counter-clockwise).
    """
    return [lateral, -down, -push, 0.0, yaw, 0.0]


def scripted_cases(steps=1500):
    """Per-env input sequences: each env is one scenario."""
    cases = {}
    t = np.arange(steps)
    cases["steady_push"] = [wrench(push=14.6, down=20.0) for _ in t]
    cases["ramp"] = [wrench(push=min(40.0, 0.2 * k), down=25.0) for k in t]
    cases["twist"] = [wrench(push=15.0, down=20.0, yaw=0.8 if k > 100 else 0.0) for k in t]
    cases["deadman_release"] = [wrench(push=15.0, down=20.0 if k < 300 else 0.0) for k in t]
    cases["collapse_spike"] = [wrench(push=15.0, down=20.0 if not 300 <= k < 310 else 400.0) for k in t]
    # The 0.2 low-pass caps the filtered drive force's slope at 0.2 * jump / dt, so the 800 N/s
    # monitor only fires on a raw jump above 80 N. 15 -> 60 N (450 N/s filtered) would not.
    cases["impulse"] = [wrench(push=15.0 if k < 300 else 120.0, down=20.0) for k in t]
    cases["light_rest"] = [wrench(push=0.0, down=2.0) for _ in t]
    rng = np.random.default_rng(7)
    cases["noisy"] = [
        wrench(
            push=12 + 6 * rng.standard_normal(),
            down=20 + 5 * rng.standard_normal(),
            lateral=3 * rng.standard_normal(),
            yaw=0.5 * rng.standard_normal(),
        )
        for _ in t
    ]
    return cases


def run_both(enforce_halts=True, lpf_alpha=0.2):
    cases = scripted_cases()
    names = list(cases)
    steps = len(cases[names[0]])
    fp = wa.ForceTorqueProcessorParams(lpf_alpha=lpf_alpha)
    cp = wa.AdmittanceParams(enforce_halts=enforce_halts)
    proc = wa.ForceTorqueProcessor(len(names), "cpu", fp)
    ctrl = wa.WalkerAdmittance(len(names), "cpu", cp)
    for state in ("offset", "filtered"):
        setattr(proc, state, getattr(proc, state).double())
    for state in ("v_x", "omega_z", "prev_f_drive", "prev_u_steer"):
        setattr(ctrl, state, getattr(ctrl, state).double())

    refs = [(ReferenceProcessor(fp), ReferenceController(cp)) for _ in names]
    port_v = np.zeros((steps, len(names)))
    port_w = np.zeros((steps, len(names)))
    port_halt = np.zeros((steps, len(names)), dtype=bool)
    ref_v = np.zeros_like(port_v)
    ref_w = np.zeros_like(port_w)
    ref_halt = np.zeros_like(port_halt)
    for k in range(steps):
        raw = torch.tensor([cases[n][k] for n in names], dtype=torch.float64)
        out = ctrl.step(proc.step(raw))
        port_v[k], port_w[k] = out["v_x"].numpy(), out["omega_z"].numpy()
        port_halt[k] = (out["halt_deadman"] | out["halt_collapse"] | out["halt_impulse"]).numpy()
        for e, n in enumerate(names):
            rp, rc = refs[e]
            ref_v[k, e], ref_w[k, e], ref_halt[k, e] = rc.step(rp.step(cases[n][k]))
    return names, port_v, port_w, port_halt, ref_v, ref_w, ref_halt


def test_port_matches_the_cpp_transcription():
    names, pv, pw, ph, rv, rw, rh = run_both()
    assert np.abs(pv - rv).max() < 1e-9, np.abs(pv - rv).max(axis=0)
    assert np.abs(pw - rw).max() < 1e-9
    assert (ph == rh).all(), [n for e, n in enumerate(names) if (ph[:, e] != rh[:, e]).any()]


def test_steady_push_reaches_force_over_damping():
    """V5: a constant 14.6 N push settles at F / B_drive = 0.2433 m/s, the reference's walking speed."""
    names, pv, *_ = run_both()
    v = pv[-1, names.index("steady_push")]
    assert abs(v - 14.6 / 60.0) < 1e-3, v


def test_yaw_torque_turns_left_and_respects_the_clamp():
    names, pv, pw, *_ = run_both()
    e = names.index("twist")
    assert pw[-1, e] > 0.0  # counter-clockwise torque, leftward yaw rate
    assert (np.abs(pw[:, e]) <= np.abs(pv[:, e]) / 0.5 + 1e-12).all()


def test_each_halt_fires_at_its_threshold_and_zeroes_the_state():
    """V7: dead-man release, collapse spike and impulse each halt, and a halt stops the walker.

    Latency is the low-pass filter's, and it is real device behaviour: after a 20 N rest is
    released the filtered load decays as 20 * 0.8^k and crosses the 0.10 N dead-man after 24
    ticks (~0.46 s). A 400 N collapse load crosses 200 N on the third tick; a 120 N impulse
    trips on the first.
    """
    names, pv, _, ph, *_ = run_both()
    for case, onset, latency in (("deadman_release", 300, 30), ("collapse_spike", 300, 5), ("impulse", 300, 2)):
        e = names.index(case)
        assert not ph[250:299, e].any(), case
        assert ph[onset : onset + latency, e].any(), case
        first = onset + int(np.argmax(ph[onset:, e]))
        assert pv[first, e] == 0.0, case
    # Released for good, the dead-man keeps the walker stopped.
    assert (pv[330:, names.index("deadman_release")] == 0.0).all()


def test_first_tick_impulse_matches_the_robot():
    """Differentiating against zero on the first tick: unfiltered, a 14.6 N push is 730 N/s and passes."""
    names, _, _, ph, *_ = run_both(lpf_alpha=1.0)
    assert not ph[0, names.index("steady_push")]


def test_idle_bias_walks_a_light_rest_into_the_deadman():
    """Real device behaviour kept on purpose: a sub-3 N rest is absorbed into the bias and trips the dead-man.

    A 2 N rest needs ~700 ticks (14 s) for the bias to pass 1 N and the deadband to zero it.
    """
    names, _, _, ph, *_ = run_both()
    e = names.index("light_rest")
    assert ph[-50:, e].all()


def test_measurement_mode_reports_halts_without_stopping():
    names, pv, _, ph, *_ = run_both(enforce_halts=False)
    e = names.index("deadman_release")
    assert ph[330:, e].all() and pv[330, e] != 0.0
