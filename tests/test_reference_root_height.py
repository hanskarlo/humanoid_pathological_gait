# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-phase reference pelvis height, exercised without Isaac Sim.

The height reward previously tracked a constant 1.05 m while the reference pelvis rises
and falls 30 mm over the stride. Narrowing that constant target's ``std`` to charge harder
for crouching cost a full 75-minute run and regressed every gait metric, so the sampler
that replaces it is worth pinning down cheaply.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

EXT = pathlib.Path(__file__).resolve().parents[1] / "source" / "humanoid_pathological_gait"
sys.path.insert(0, str(EXT))
DATA = EXT / "humanoid_pathological_gait/tasks/humanoid_pathological_gait/data/retargeted_h1_stride.npz"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="reference stride not staged")


@pytest.fixture(scope="module")
def manager():
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.h1_joints import (
        CLINICAL_JOINT_ORDER,
        H1JointLayout,
    )
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.reference import ReferenceGaitManager

    layout = H1JointLayout.from_sim_names(list(CLINICAL_JOINT_ORDER), device="cpu")
    return ReferenceGaitManager(str(DATA), layout, num_envs=8, device="cpu")


def test_height_table_is_loaded_and_physically_plausible(manager):
    assert manager.ref_root_height is not None, "root_translation z was dropped again"
    assert manager.ref_root_height.shape == (manager.num_samples,)
    assert torch.all(manager.ref_root_height > 0.8)
    assert torch.all(manager.ref_root_height < 1.3)


def test_pelvis_actually_oscillates(manager):
    """The whole reason for this feature: the reference is not at a constant height."""
    span = float(manager.ref_root_height.max() - manager.ref_root_height.min())
    assert span > 0.015, f"expected a real vertical excursion, got {span * 1000:.1f} mm"


def test_sampling_follows_phase(manager):
    """Distinct phases give distinct heights, and phase 0 matches the table's first entry."""
    manager.gait_phase[:] = torch.linspace(0.0, 1.0, manager.num_envs)
    sampled = manager.sample_root_height()

    assert sampled.shape == (manager.num_envs,)
    assert torch.isfinite(sampled).all()
    torch.testing.assert_close(sampled[0], manager.ref_root_height[0])
    torch.testing.assert_close(sampled[-1], manager.ref_root_height[-1])
    assert sampled.std() > 0.0, "sampler returned a constant; phase is being ignored"


def test_sampled_range_matches_the_table(manager):
    """Sweeping every phase reproduces the table's own span, so nothing is clipped."""
    manager.gait_phase[:] = 0.0
    seen = []
    for phase in torch.linspace(0.0, 1.0, 200):
        manager.gait_phase[:] = phase
        seen.append(manager.sample_root_height()[0])
    seen = torch.stack(seen)

    assert seen.min() >= manager.ref_root_height.min() - 1e-6
    assert seen.max() <= manager.ref_root_height.max() + 1e-6
    span_ratio = (seen.max() - seen.min()) / (manager.ref_root_height.max() - manager.ref_root_height.min())
    assert span_ratio > 0.9, f"sampler flattens the trajectory (span ratio {span_ratio:.2f})"


def test_interpolation_is_continuous(manager):
    """Nearest-neighbour on 1001 samples would staircase a 30 mm signal; lerp must not."""
    manager.gait_phase[:] = 0.0
    steps = []
    previous = None
    for phase in torch.linspace(0.3, 0.4, 400):
        manager.gait_phase[:] = phase
        value = float(manager.sample_root_height()[0])
        if previous is not None:
            steps.append(abs(value - previous))
        previous = value

    # No single sub-step may jump more than a fifth of the whole excursion.
    span = float(manager.ref_root_height.max() - manager.ref_root_height.min())
    assert max(steps) < 0.2 * span


def test_subset_indexing(manager):
    """Passing env_ids selects those environments, not the whole batch."""
    manager.gait_phase[:] = torch.linspace(0.0, 1.0, manager.num_envs)
    ids = torch.tensor([1, 3, 5])
    subset = manager.sample_root_height(ids)
    everything = manager.sample_root_height()

    assert subset.shape == (3,)
    torch.testing.assert_close(subset, everything[ids])
