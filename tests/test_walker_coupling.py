# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grip coupling, the sensor-equivalent wrench, and per-hand reconstruction -- signs first.

Every check here is a physical statement a device engineer would make about the real walker:
pushing forward reads as positive drive, leaning down as positive load, and pushing harder with
the left hand turns the walker right. They are checked against the real controller port, so a
sign error anywhere in the chain fails here and not in a rollout.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

PKG = (
    pathlib.Path(__file__).resolve().parents[1]
    / "source/humanoid_pathological_gait/humanoid_pathological_gait/tasks/humanoid_pathological_gait"
)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PKG / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wc = _load("walker_coupling")
wa = _load("walker_admittance")

D, H = 0.20, 0.0825  # half grip width; bar offset behind the sensor (asset: handle x=-0.395, sensor x=-0.3125)
IDENTITY = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)


U = 0.031  # grips above the sensor axis (handle fitted to the H1's hands)


def grips(quat=IDENTITY, up=U):
    """Grip and sensor positions for a walker at the origin with orientation ``quat``."""
    # Body frame: sensor at the origin, bar H behind it (-x_b), left grip at +y_b, grips `up` above.
    body = torch.tensor([[[-H, D, up], [-H, -D, up]]], dtype=torch.float64)
    x, y, z, w = quat[0]
    # rotate body -> world
    qv = torch.tensor([x, y, z], dtype=torch.float64)
    t = 2.0 * torch.cross(qv.expand_as(body), body, dim=-1)
    world = body + w * t + torch.cross(qv.expand_as(body), t, dim=-1)
    return world, torch.zeros(1, 3, dtype=torch.float64)


def world_force(body_force, quat=IDENTITY):
    x, y, z, w = quat[0]
    qv = torch.tensor([x, y, z], dtype=torch.float64)
    t = 2.0 * torch.cross(qv.expand_as(body_force), body_force, dim=-1)
    return body_force + w * t + torch.cross(qv.expand_as(body_force), t, dim=-1)


def wrench_for(left_body, right_body, quat=IDENTITY):
    grip, sensor = grips(quat)
    force_b = torch.tensor([[left_body, right_body]], dtype=torch.float64)
    return wc.sensor_wrench(world_force(force_b, quat), grip, sensor, quat)


def test_the_grip_spring_is_equal_and_opposite():
    p_hand = torch.zeros(1, 2, 3)
    p_grip = torch.tensor([[[0.01, 0.0, 0.0], [0.0, -0.02, 0.0]]])
    f = wc.grip_spring_forces(p_hand, torch.zeros_like(p_hand), p_grip, torch.zeros_like(p_hand), 5000.0, 0.0)
    assert torch.allclose(f, 5000.0 * p_grip)


def test_pushing_forward_reads_as_positive_drive_and_moves_the_walker_forward():
    w = wrench_for([15.0, 0.0, 0.0], [15.0, 0.0, 0.0])  # body +x is forward
    assert w[0, 2] < 0  # F_z (z_p backward) negative ...
    ctrl = wa.WalkerAdmittance(1, "cpu", wa.AdmittanceParams(enforce_halts=False))
    for state in ("v_x", "omega_z", "prev_f_drive", "prev_u_steer"):
        setattr(ctrl, state, getattr(ctrl, state).double())
    out = ctrl.step(w)
    assert out["f_drive"].item() == pytest.approx(30.0)  # ... so f_drive = -F_z is positive
    assert out["v_x"].item() > 0


def test_leaning_down_reads_as_positive_load():
    w = wrench_for([0.0, 0.0, -40.0], [0.0, 0.0, -40.0])  # body -z is down
    assert -w[0, 1].item() == pytest.approx(80.0)  # f_down = -F_y


def test_pushing_harder_with_the_left_hand_turns_the_walker_right():
    """Physical statement: more forward push on the left grip yaws the walker clockwise (right)."""
    w = wrench_for([25.0, 0.0, 0.0], [5.0, 0.0, 0.0])
    assert w[0, 4] < 0  # T_y about up: clockwise
    ctrl = wa.WalkerAdmittance(1, "cpu", wa.AdmittanceParams(enforce_halts=False))
    for state in ("v_x", "omega_z", "prev_f_drive", "prev_u_steer"):
        setattr(ctrl, state, getattr(ctrl, state).double())
    for _ in range(50):
        out = ctrl.step(w)
    assert out["omega_z"].item() < 0  # the real controller's law turns it right


def test_reconstruction_recovers_each_hand_exactly():
    """W3: down and drive per hand come back from one sensor, lateral forces in any split."""
    torch.manual_seed(0)
    for _ in range(20):
        left = (torch.randn(3, dtype=torch.float64) * 20).tolist()
        right = (torch.randn(3, dtype=torch.float64) * 20).tolist()
        w = wrench_for(left, right)
        r = wc.reconstruct_hand_loads(w, D, H, U)
        # processed-frame user forces: down = body +z pushed down -> f_down = -(body z) ...
        assert r["down_left"].item() == pytest.approx(-left[2], abs=1e-9)
        assert r["down_right"].item() == pytest.approx(-right[2], abs=1e-9)
        assert r["drive_left"].item() == pytest.approx(left[0], abs=1e-9)
        assert r["drive_right"].item() == pytest.approx(right[0], abs=1e-9)
        assert abs(r["tx_residual"].item()) < 1e-9


def test_the_reading_does_not_depend_on_which_way_the_walker_faces():
    """The sensor is fixed to the walker, so a yawed walker with the same body-frame push reads the same."""
    yaw = torch.tensor(0.9, dtype=torch.float64)
    quat = torch.stack([torch.zeros(()), torch.zeros(()), torch.sin(yaw / 2), torch.cos(yaw / 2)]).double().unsqueeze(0)
    a = wrench_for([20.0, 3.0, -30.0], [10.0, -2.0, -50.0])
    b = wrench_for([20.0, 3.0, -30.0], [10.0, -2.0, -50.0], quat)
    assert torch.allclose(a, b, atol=1e-9)


def test_asymmetry_index():
    assert wc.asymmetry_index(torch.tensor(30.0), torch.tensor(10.0)).item() == pytest.approx(0.5)


def test_ignoring_the_grip_height_biases_the_vertical_split():
    """The reason grip_up exists: with a lateral push, leaving it out misattributes vertical load."""
    w = wrench_for([10.0, 15.0, -40.0], [10.0, 15.0, -40.0])  # symmetric down load, sideways push
    right = wc.reconstruct_hand_loads(w, D, H, U)
    wrong = wc.reconstruct_hand_loads(w, D, H, 0.0)
    assert right["down_left"].item() == pytest.approx(right["down_right"].item(), abs=1e-9)
    assert abs(wrong["down_left"].item() - wrong["down_right"].item()) > 1.0
