# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H1JointLayout.scaled_tracking_weights (the --frontal_freedom hip-roll relaxation), without Isaac Sim."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

PATH = pathlib.Path(__file__).resolve().parents[1] / (
    "source/humanoid_pathological_gait/humanoid_pathological_gait/tasks/humanoid_pathological_gait/h1_joints.py"
)


@pytest.fixture(scope="module")
def layout():
    spec = importlib.util.spec_from_file_location("h1_joints_under_test", PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["h1_joints_under_test"] = module
    spec.loader.exec_module(module)
    return module.H1JointLayout.from_sim_names(list(module.CLINICAL_JOINT_ORDER)[::-1], device="cpu")


def test_none_is_the_default_tensor_itself(layout):
    assert layout.scaled_tracking_weights(None) is layout.tracking_weights


def test_only_the_named_joints_change(layout):
    scaled = layout.scaled_tracking_weights({"hip_roll": 0.2})
    for i, name in enumerate(layout.sim_names):
        expected = layout.tracking_weights[i] * (0.2 if name.endswith("hip_roll") else 1.0)
        assert torch.isclose(scaled[i], expected), name
    assert int((scaled != layout.tracking_weights).sum()) == 2  # left and right hip roll
