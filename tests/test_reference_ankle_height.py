# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-phase reference ankle height (``track_ankle_height``'s table), exercised without Isaac Sim."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

EXT = pathlib.Path(__file__).resolve().parents[1] / "source" / "humanoid_pathological_gait"
sys.path.insert(0, str(EXT))
DATA = EXT / "humanoid_pathological_gait/tasks/humanoid_pathological_gait/data"
ANKLES = DATA / "references/healthy_s049_stride0_grounded_ankles.npz"
STAGED = DATA / "retargeted_h1_stride.npz"

pytestmark = pytest.mark.skipif(not ANKLES.exists(), reason="ankle archive not staged")


def _manager(path: pathlib.Path):
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.h1_joints import (
        CLINICAL_JOINT_ORDER,
        H1JointLayout,
    )
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.reference import ReferenceGaitManager

    layout = H1JointLayout.from_sim_names(list(CLINICAL_JOINT_ORDER), device="cpu")
    return ReferenceGaitManager(str(path), layout, num_envs=8, device="cpu")


def test_table_is_loaded_in_paretic_sound_order():
    manager = _manager(ANKLES)
    with np.load(ANKLES, allow_pickle=True) as archive:
        raw = np.asarray(archive["reference_ankle_height"])
        side = str(archive["impaired_side"])
    expected = raw if side == "left" else raw[:, ::-1]
    assert manager.ref_ankle_height.shape == (manager.num_samples, 2)
    assert np.allclose(manager.ref_ankle_height.numpy(), expected)
    # A flat foot puts the ankle origin ~0.070 m up; swing lifts it well above that.
    assert 0.06 < float(manager.ref_ankle_height.min()) < 0.08
    assert float(manager.ref_ankle_height.max()) > 0.15


def test_sampling_interpolates_and_hits_the_table_at_its_nodes():
    manager = _manager(ANKLES)
    n = manager.num_samples
    manager.gait_phase[:] = torch.tensor([0.0, 10 / (n - 1), 10.5 / (n - 1), 0.5, 0.25, 0.75, 0.9, 0.999])
    sampled = manager.sample_ankle_height()
    table = manager.ref_ankle_height
    assert sampled.shape == (8, 2)
    assert torch.allclose(sampled[0], table[0])
    assert torch.allclose(sampled[1], table[10])
    assert torch.allclose(sampled[2], 0.5 * (table[10] + table[11]), atol=1e-6)


def test_an_archive_without_the_table_gives_none():
    if not STAGED.exists():
        pytest.skip("staged stride not present")
    manager = _manager(STAGED)
    assert manager.ref_ankle_height is None
    assert manager.sample_ankle_height() is None
