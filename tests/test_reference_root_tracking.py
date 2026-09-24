# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Root-progression tracking, exercised without Isaac Sim.

``ReferenceGaitManager`` needs only torch and the staged stride archive, so the anchoring
arithmetic can be tested directly. It is worth testing directly: the first full-scale run
of this feature regressed on every metric because the reference displacement was measured
from phase 0 while the anchor was laid at whatever phase the environment reset to.
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
    return ReferenceGaitManager(str(DATA), layout, num_envs=4, device="cpu")


def test_the_archive_supplies_a_root_trajectory(manager):
    assert manager.ref_root_disp is not None, "archive carries no root_translation"
    assert manager.ref_root_disp.shape == (manager.num_samples, 2)
    # Rotated into the stride's own heading, so it ends up travelling along +x.
    assert manager.ref_root_disp[-1, 0] > 0.2
    assert abs(float(manager.ref_root_disp[-1, 1])) < 0.05


def test_an_anchor_laid_mid_stride_starts_with_no_error(manager):
    """The regression that cost a 1.3-hour run.

    Training randomises the start phase and ``evaluate.py`` spreads phases across
    environments, so anchors are routinely laid mid-stride. Measuring the target from phase
    0 charged an environment resetting at phase 0.75 with 0.34 m of debt it could not repay.
    """
    zeros, yaw = torch.zeros(4, 2), torch.zeros(4)
    manager.gait_phase[:] = torch.tensor([0.0, 0.25, 0.50, 0.75])
    manager.set_cycle_anchor(zeros, yaw)
    error = manager.root_progression_error(zeros, yaw)
    assert torch.allclose(error, torch.zeros_like(error), atol=1e-6)


def _reference_travel(manager):
    index = torch.round(manager.gait_phase * (manager.num_samples - 1)).long()
    anchor = torch.round(manager.cycle_anchor_phase * (manager.num_samples - 1)).long()
    return manager.ref_root_disp[index] - manager.ref_root_disp[anchor]


def test_matching_the_reference_leaves_no_error(manager):
    """Walking the stride -- mirrored for a right-paretic environment -- scores zero error."""
    zeros, yaw = torch.zeros(4, 2), torch.zeros(4)
    manager.paretic_side[:] = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    manager.gait_phase[:] = torch.tensor([0.10, 0.30, 0.50, 0.70])
    manager.set_cycle_anchor(zeros, yaw)

    manager.gait_phase[:] = torch.tensor([0.35, 0.55, 0.75, 0.95])
    travelled = _reference_travel(manager)
    right = manager.paretic_side > 0
    travelled[right, 1] = -travelled[right, 1]

    error = manager.root_progression_error(travelled, yaw)
    assert torch.allclose(error, torch.zeros_like(error), atol=1e-5)


def test_a_right_paretic_environment_is_asked_for_the_mirrored_sway(manager):
    """The regression: every right-paretic environment was scored against the unmirrored sway.

    Following the *unmirrored* path in a right-paretic environment must now read as a
    cross-track error of twice the reference's lateral travel, and along-track must be
    untouched -- the mirror is a reflection across the direction of travel.
    """
    zeros, yaw = torch.zeros(4, 2), torch.zeros(4)
    manager.paretic_side[:] = torch.ones(4)
    manager.gait_phase[:] = torch.zeros(4)
    manager.set_cycle_anchor(zeros, yaw)
    # Phases where the reference has swayed well off its line of travel.
    lateral = manager.ref_root_disp[:, 1]
    peak = float(lateral.abs().argmax()) / (manager.num_samples - 1)
    manager.gait_phase[:] = torch.full((4,), peak)
    unmirrored = _reference_travel(manager)

    error = manager.root_progression_error(unmirrored, yaw)
    assert torch.allclose(error[:, 0], torch.zeros(4), atol=1e-5)
    assert torch.allclose(error[:, 1], 2.0 * unmirrored[:, 1], atol=1e-5)
    assert float(unmirrored[:, 1].abs().min()) > 0.05, "the stride should sway more than 5 cm at its peak"


def test_standing_still_falls_behind(manager):
    """Sign convention: negative along-track means behind the reference."""
    zeros, yaw = torch.zeros(4, 2), torch.zeros(4)
    manager.gait_phase[:] = torch.zeros(4)
    manager.set_cycle_anchor(zeros, yaw)
    manager.gait_phase[:] = torch.full((4,), 0.5)
    error = manager.root_progression_error(zeros, yaw)
    assert (error[:, 0] < -0.1).all(), "a stationary robot should read as behind"


def test_error_is_measured_in_the_anchor_heading(manager):
    """A robot facing backwards that walks its own +x is going the wrong way."""
    yaw_zero, yaw_pi = torch.zeros(4), torch.full((4,), float(torch.pi))
    zeros = torch.zeros(4, 2)
    manager.gait_phase[:] = torch.zeros(4)

    manager.set_cycle_anchor(zeros, yaw_zero)
    manager.gait_phase[:] = torch.full((4,), 0.5)
    forward = manager.root_progression_error(torch.tensor([[0.2, 0.0]] * 4), yaw_zero)

    manager.gait_phase[:] = torch.zeros(4)
    manager.set_cycle_anchor(zeros, yaw_pi)
    manager.gait_phase[:] = torch.full((4,), 0.5)
    reversed_heading = manager.root_progression_error(torch.tensor([[0.2, 0.0]] * 4), yaw_pi)

    assert (forward[:, 0] > reversed_heading[:, 0]).all()


def test_wrapping_the_phase_is_flagged_for_re_anchoring(manager):
    manager.gait_phase[:] = torch.tensor([0.1, 0.5, 0.9, 0.99])
    manager.advance(dt=0.2, rate_scale=1.0)
    assert manager.cycle_wrapped.shape == (4,)
    assert manager.cycle_wrapped.dtype == torch.bool
