# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TSRT reflex thresholds, checked against the reference stride's own kinematics.

A reflex threshold placed outside the reference's range of motion cannot shape the gait --
it can only brake gross deviation after the fact. That is what happened to the knee: the
threshold sat at 20.05 deg while the reference paretic knee peaks at 14.80 deg, so the
stiff-knee pathology never emerged and a trained policy flexed to 40 deg. These tests pin
the threshold into the functional range so the placement cannot silently regress.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

EXT = pathlib.Path(__file__).resolve().parents[1] / "source" / "humanoid_pathological_gait"
sys.path.insert(0, str(EXT))
DATA = EXT / "humanoid_pathological_gait/tasks/humanoid_pathological_gait/data/retargeted_h1_stride.npz"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="reference stride not staged")


@pytest.fixture(scope="module")
def stride():
    archive = np.load(DATA)
    names = [str(n) for n in archive["joint_names"]]
    side = str(archive["impaired_side"])
    return archive["q_trajectory"], names, side


@pytest.fixture(scope="module")
def params():
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.tsrt import TSRTParams

    return TSRTParams()


def _paretic(stride, joint):
    q, names, side = stride
    return q[:, names.index(f"{side}_{joint}")]


def test_knee_threshold_sits_inside_the_functional_range(stride, params):
    """The reflex must be reachable by the motion it is supposed to shape."""
    knee = _paretic(stride, "knee")
    crossed = float(np.mean(knee > params.lambda_0_knee))

    assert crossed > 0.0, (
        f"lambda_0_knee={params.lambda_0_knee} rad ({np.degrees(params.lambda_0_knee):.2f} deg) is above the "
        f"reference's peak paretic knee flexion ({np.degrees(knee.max()):.2f} deg): the reflex can never "
        "shape the stride, only brake deviation from it"
    )


def test_knee_threshold_does_not_swamp_the_reference(stride, params):
    """...but not so low that it fights the motion the policy is asked to reproduce."""
    knee = _paretic(stride, "knee")
    crossed = float(np.mean(knee > params.lambda_0_knee))

    assert crossed < 0.25, (
        f"the reference crosses lambda_0_knee on {crossed:.1%} of the stride; the reflex would resist "
        "reference motion itself, not just excess flexion"
    )


def test_ankle_threshold_is_engaged_by_the_reference(stride, params):
    """Foot drop is the one pathology that already reproduces; keep its threshold reachable.

    The ankle's stretch coordinate is negated (plantarflexors resist dorsiflexion), so the
    comparison runs the other way round from the knee's.
    """
    ankle = _paretic(stride, "ankle")
    crossed = float(np.mean(-ankle > params.lambda_0_ankle))

    assert crossed > 0.5, f"ankle reflex engaged on only {crossed:.1%} of the reference stride"


def test_reflex_opposes_flexion_past_the_threshold(params):
    """Sign check: past threshold and lengthening, the knee torque must resist."""
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.h1_joints import (
        CLINICAL_JOINT_ORDER,
        H1JointLayout,
    )
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.tsrt import TSRTSpasticModel

    layout = H1JointLayout.from_sim_names(list(CLINICAL_JOINT_ORDER), device="cpu")
    model = TSRTSpasticModel(layout, params, device="cpu")
    knee_id = layout.index_of("left_knee")

    pos = torch.zeros(1, len(CLINICAL_JOINT_ORDER))
    vel = torch.zeros_like(pos)
    pos[0, knee_id] = params.lambda_0_knee + 0.30
    vel[0, knee_id] = 1.0
    left_paretic = torch.tensor([-1.0])

    tau = model.compute(pos, vel, left_paretic)
    assert tau[0, knee_id] < 0.0, "reflex should oppose flexion, not assist it"

    # Below the deadband the reflex stays silent even when past the threshold.
    vel[0, knee_id] = 0.5 * params.velocity_deadband
    quiet = model.compute(pos, vel, left_paretic)
    assert quiet[0, knee_id] == 0.0


def test_lowering_the_threshold_strengthens_resistance(params):
    """The change is only meaningful if it actually increases torque at a given angle."""
    from dataclasses import replace

    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.h1_joints import (
        CLINICAL_JOINT_ORDER,
        H1JointLayout,
    )
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.tsrt import TSRTSpasticModel

    layout = H1JointLayout.from_sim_names(list(CLINICAL_JOINT_ORDER), device="cpu")
    knee_id = layout.index_of("left_knee")

    pos = torch.zeros(1, len(CLINICAL_JOINT_ORDER))
    vel = torch.zeros_like(pos)
    pos[0, knee_id] = 0.70  # 40 deg, what an untamed policy actually reached
    vel[0, knee_id] = 1.0
    left_paretic = torch.tensor([-1.0])

    now = TSRTSpasticModel(layout, params, device="cpu").compute(pos, vel, left_paretic)[0, knee_id]
    before = TSRTSpasticModel(layout, replace(params, lambda_0_knee=0.35), device="cpu").compute(
        pos, vel, left_paretic
    )[0, knee_id]

    assert abs(float(now)) > abs(float(before))
