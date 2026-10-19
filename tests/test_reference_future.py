# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference sampled ahead of the current phase (the ``reference_future`` observation), without Isaac Sim."""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

EXT = pathlib.Path(__file__).resolve().parents[1] / "source" / "humanoid_pathological_gait"
sys.path.insert(0, str(EXT))
STRIDE = (
    EXT
    / "humanoid_pathological_gait/tasks/humanoid_pathological_gait/data/references/healthy_s049_stride0_grounded.npz"
)

pytestmark = pytest.mark.skipif(not STRIDE.exists(), reason="grounded healthy stride not staged")


@pytest.fixture()
def manager():
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.h1_joints import (
        CLINICAL_JOINT_ORDER,
        H1JointLayout,
    )
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.reference import ReferenceGaitManager

    layout = H1JointLayout.from_sim_names(list(CLINICAL_JOINT_ORDER), device="cpu")
    m = ReferenceGaitManager(str(STRIDE), layout, num_envs=8, device="cpu")
    m.gait_phase[:] = torch.linspace(0.0, 0.95, 8)
    m.paretic_side[:] = torch.tensor([-1.0, 1.0] * 4)  # both sides, so the mirroring is exercised
    return m


def test_zero_lead_is_the_current_sample(manager):
    q_now, _ = manager.sample()
    q_ahead, contact = manager.sample_ahead(0.0)
    assert torch.allclose(q_ahead, q_now)
    assert torch.equal(contact, manager.sample_contact())


def test_a_whole_stride_ahead_wraps_to_the_same_pose(manager):
    q_now, _ = manager.sample()
    q_ahead, _ = manager.sample_ahead(manager.stride_duration)
    assert torch.allclose(q_ahead, q_now, atol=1e-5)


def test_lead_matches_the_sample_at_the_shifted_phase(manager):
    lead = 0.2
    q_ahead, contact_ahead = manager.sample_ahead(lead)
    phase = manager.gait_phase.clone()
    manager.gait_phase[:] = torch.remainder(phase + lead / manager.stride_duration, 1.0)
    q_shifted, _ = manager.sample()
    contact_shifted = manager.sample_contact()
    manager.gait_phase[:] = phase
    assert torch.allclose(q_ahead, q_shifted)
    assert torch.equal(contact_ahead, contact_shifted)
    assert not torch.allclose(q_ahead, manager.sample()[0]), "the lead must actually change the target"
