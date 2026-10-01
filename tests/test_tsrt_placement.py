# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-patient reflex-threshold placement (H4): it must be subject 0's rule, not a new one.

Three properties: placing subject 0 against itself returns subject 0's hand-placed thresholds;
a right-impaired mirror of subject 0 places identically (the side bookkeeping is right); and a
new patient's thresholds are crossed on the calibration fraction of its own stride.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

EXT = pathlib.Path(__file__).resolve().parents[1] / "source" / "humanoid_pathological_gait"
sys.path.insert(0, str(EXT))
DATA = EXT / "humanoid_pathological_gait/tasks/humanoid_pathological_gait/data"
STAGED = DATA / "retargeted_h1_stride.npz"
SUBJECT3 = DATA / "references/subject03_stride0.npz"

pytestmark = pytest.mark.skipif(not STAGED.exists(), reason="reference stride not staged")


def _placement(path):
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.tsrt import TSRTParams, place_reflex_thresholds

    return place_reflex_thresholds(path, STAGED, TSRTParams()), TSRTParams()


def test_subject_zero_reproduces_its_own_thresholds():
    placed, params = _placement(STAGED)
    assert abs(placed["knee"]["lambda_0"] - params.lambda_0_knee) < 1e-3
    assert abs(placed["hip_roll"]["lambda_0"] - params.lambda_0_hip) < 1e-3
    # The fractions the 09-07/08 entries quote.
    assert round(placed["knee"]["calibration_fraction"], 3) == 0.038
    assert round(placed["hip_roll"]["calibration_fraction"], 3) == 0.044


def test_a_right_impaired_mirror_places_identically(tmp_path):
    import torch

    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.h1_joints import (
        CLINICAL_JOINT_ORDER,
        H1JointLayout,
    )

    layout = H1JointLayout.from_sim_names(list(CLINICAL_JOINT_ORDER), device="cpu")
    with np.load(STAGED, allow_pickle=True) as z:
        archive = {k: z[k] for k in z.files}
    archive["q_trajectory"] = layout.mirror(torch.tensor(archive["q_trajectory"], dtype=torch.float32)).numpy()
    archive["impaired_side"] = np.array("right")
    path = tmp_path / "mirror.npz"
    np.savez(path, **archive)
    left, _ = _placement(STAGED)
    right, _ = _placement(path)
    for joint in ("knee", "hip_roll"):
        assert abs(left[joint]["lambda_0"] - right[joint]["lambda_0"]) < 1e-5, joint


@pytest.mark.skipif(not SUBJECT3.exists(), reason="subject 3 reference not staged")
def test_a_new_patient_is_crossed_on_the_calibration_fraction():
    placed, params = _placement(SUBJECT3)
    for joint in ("knee", "hip_roll"):
        p = placed[joint]
        assert abs(p["crossing_fraction"] - p["calibration_fraction"]) < 0.005, joint
    # The reason the rule exists: subject 0's knee threshold sits inside subject 3's range throughout.
    assert placed["knee"]["unplaced_crossing_fraction"] > 0.99
    assert placed["knee"]["lambda_0"] > params.lambda_0_knee
