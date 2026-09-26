# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A right-impaired stride must load into exactly the tables its left-impaired mirror does.

Every run before H4 used subject 0, whose impaired side is the left, so the right-impaired load
path had never executed. It carried two defects (2026-09-24 audit): per-joint ROM computed before
the right-impaired table swap, handing the sound limb's tracking widths to the paretic limb, and
root-state tables left in the archive's frame while the joints were swapped into the canonical
paretic-left one.

The test needs no second patient. It builds the exact sagittal mirror of the staged left-impaired
archive and labels it right-impaired: a patient who is the mirror image of subject 0. The
reference manager canonicalises to "paretic limb on the left", so the two must produce identical
tables -- joints, per-joint ROM, contact schedule, root displacement, root orientation and
velocities.
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
def layout():
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.h1_joints import (
        CLINICAL_JOINT_ORDER,
        H1JointLayout,
    )

    # Simulation order == clinical order here, so layout.mirror acts on the archive's own columns.
    return H1JointLayout.from_sim_names(list(CLINICAL_JOINT_ORDER), device="cpu")


@pytest.fixture(scope="module")
def managers(layout, tmp_path_factory):
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.reference import ReferenceGaitManager

    with np.load(DATA, allow_pickle=True) as z:
        archive = {k: z[k] for k in z.files}
    assert str(archive["impaired_side"]) == "left", "the staged archive is expected to be left-impaired"

    mirrored = dict(archive)
    for key in ("q_trajectory", "v_trajectory"):
        mirrored[key] = layout.mirror(torch.tensor(archive[key], dtype=torch.float32)).numpy()
    mirrored["impaired_side"] = np.array("right")
    mirrored["reference_contact"] = archive["reference_contact"][:, ::-1].copy()  # stored (left, right)
    mirrored["swing_fraction_left"] = archive["swing_fraction_right"]
    mirrored["swing_fraction_right"] = archive["swing_fraction_left"]
    translation = archive["root_translation"].copy()
    translation[:, 1] *= -1.0  # reflection across the sagittal (x-z) plane
    mirrored["root_translation"] = translation
    quaternion = archive["root_quaternion"].copy()  # (w, x, y, z): vector part is a pseudovector
    quaternion[:, 1] *= -1.0
    quaternion[:, 3] *= -1.0
    mirrored["root_quaternion"] = quaternion
    # reference_mos is mirror-invariant (min of the two edge distances), so it is kept as is.

    path = tmp_path_factory.mktemp("mirror") / "retargeted_h1_stride_right.npz"
    np.savez(path, **mirrored)
    left = ReferenceGaitManager(str(DATA), layout, num_envs=2, device="cpu")
    right = ReferenceGaitManager(str(path), layout, num_envs=2, device="cpu")
    return left, right


@pytest.mark.parametrize(
    "table",
    [
        "ref_q",
        "ref_q_mirrored",
        "ref_v",
        "ref_joint_rom",
        "ref_joint_rom_mirrored",
        "ref_contact",
        "ref_root_disp",
        "ref_root_height",
        "ref_root_quat",
        "ref_root_quat_mirrored",
        "ref_root_lin_vel",
        "ref_root_lin_vel_mirrored",
        "ref_root_ang_vel",
        "ref_root_ang_vel_mirrored",
    ],
)
def test_mirror_patient_loads_into_the_same_canonical_table(managers, table):
    left, right = managers
    a, b = getattr(left, table), getattr(right, table)
    assert a is not None and b is not None, f"{table} missing"
    assert torch.allclose(a, b, atol=1e-5), f"{table}: max |diff| {float((a - b).abs().max()):.3e}"


def test_the_widths_go_to_the_paretic_limb(managers, layout):
    """The defect in plain terms: the paretic knee's ROM must be the small one in both."""
    left, right = managers
    knee_paretic = layout.sim_names.index("left_knee")
    knee_sound = layout.sim_names.index("right_knee")
    for manager in (left, right):
        assert manager.ref_joint_rom[knee_paretic] < manager.ref_joint_rom[knee_sound]
