# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Full reference root state -- orientation, linear and angular velocity -- without Isaac Sim.

Reference-state initialisation used to place the joints on the reference and the floating
base on the articulation default: zero forward velocity against the reference's +0.244 m/s,
and a level pelvis against a reference carrying up to 9.65 deg of coronal obliquity. The
tables that fix it cross three conventions that this project has already got wrong once
each -- the archive's ``(w, x, y, z)`` against the simulator's ``(x, y, z, w)``, the stride
heading frame, and the sagittal reflection under which angular velocity is a pseudovector.
Each is pinned here against a number measured independently of the code being tested.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
numpy = pytest.importorskip("numpy")

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


@pytest.fixture(scope="module")
def archive():
    return numpy.load(str(DATA), allow_pickle=True)


def roll_degrees(quaternion: torch.Tensor) -> torch.Tensor:
    """Coronal roll from a simulation-order ``(x, y, z, w)`` quaternion."""
    x, y, z, w = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    return torch.rad2deg(torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))


def test_tables_are_populated(manager):
    assert manager.ref_root_quat is not None, "the staged archive carries root_quaternion"
    assert manager.ref_root_quat.shape == (manager.num_samples, 4)
    assert manager.ref_root_lin_vel.shape == (manager.num_samples, 3)
    assert manager.ref_root_ang_vel.shape == (manager.num_samples, 3)
    norms = manager.ref_root_quat.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "stored quaternions are unit"


def test_stored_quaternion_reproduces_the_measured_pelvic_obliquity(manager, archive):
    """The ordering check: ``(w, x, y, z)`` in, ``(x, y, z, w)`` out.

    Read with the wrong convention the same data gives a roll near 175 deg rather than near
    4, so this single assertion catches a swap in either direction. The target numbers --
    +4.25 deg in paretic stance, +8.75 in paretic swing, a +4.50 deg hiking signature -- were
    measured from the archive directly and are the ones ``gait_analysis.pelvic_obliquity``
    reports for the reference.
    """
    contact = numpy.asarray(archive["reference_contact"])
    paretic_column = 0 if str(archive["impaired_side"]) == "left" else 1
    stance = torch.tensor(contact[:, paretic_column])
    swing = ~stance

    roll = roll_degrees(manager.ref_root_quat)
    assert roll[stance].mean().item() == pytest.approx(4.25, abs=0.05)
    assert roll[swing].mean().item() == pytest.approx(8.75, abs=0.05)
    assert (roll[swing].mean() - roll[stance].mean()).item() == pytest.approx(4.50, abs=0.05)


def test_heading_frame_removes_the_strides_own_yaw(manager):
    """Forward velocity must land on +x, or a reset launches the robot sideways.

    The capture's travel direction is not the simulator's, and it is not the pelvis's own
    facing either -- the reference yaw drifts about 8 deg across the stride. Rotating by the
    displacement heading is what makes ``ref_root_lin_vel`` composable with a random reset
    yaw, and the test of it is that mean lateral velocity vanishes while mean forward
    velocity equals the archive's own recorded walking speed.
    """
    mean_velocity = manager.ref_root_lin_vel.mean(dim=0)
    assert mean_velocity[0].item() == pytest.approx(0.2442, abs=0.005), "forward = archive speed"
    assert abs(mean_velocity[1].item()) < 0.005, "no net lateral drift in the heading frame"
    assert abs(mean_velocity[2].item()) < 0.005, "a stride returns to its own height"


def test_forward_velocity_is_never_backward(manager):
    """The state RSI used to skip. Zero-mean is not a noisy version of this."""
    forward = manager.ref_root_lin_vel[:, 0]
    assert forward.min().item() > 0.0, "the reference never walks backwards"
    assert forward.max().item() < 1.0, "nor is any frame a finite-difference spike"


def test_angular_velocity_has_no_wraparound_spikes(manager):
    """A sign flip between adjacent archive quaternions would fabricate a 2/dt spike."""
    assert manager.ref_root_ang_vel.abs().max().item() < 5.0


def test_mirroring_follows_the_reflection_rules(manager):
    """Vectors negate laterally; pseudovectors negate everywhere *but* laterally.

    Angular velocity is axial, so under a sagittal reflection roll and yaw negate and pitch
    does not -- the opposite pattern to linear velocity. A quaternion's vector part obeys the
    pseudovector rule and its scalar part is invariant. Mixing these up reverses the pelvic
    obliquity on half the environments, which is how the AMP corpus cancelled 85% of its own
    asymmetry.
    """
    assert torch.allclose(manager.ref_root_lin_vel_mirrored[:, 0], manager.ref_root_lin_vel[:, 0])
    assert torch.allclose(manager.ref_root_lin_vel_mirrored[:, 1], -manager.ref_root_lin_vel[:, 1])
    assert torch.allclose(manager.ref_root_lin_vel_mirrored[:, 2], manager.ref_root_lin_vel[:, 2])

    assert torch.allclose(manager.ref_root_ang_vel_mirrored[:, 0], -manager.ref_root_ang_vel[:, 0])
    assert torch.allclose(manager.ref_root_ang_vel_mirrored[:, 1], manager.ref_root_ang_vel[:, 1])
    assert torch.allclose(manager.ref_root_ang_vel_mirrored[:, 2], -manager.ref_root_ang_vel[:, 2])

    # The reflected stride hikes the other way, by the same amount.
    assert torch.allclose(
        roll_degrees(manager.ref_root_quat_mirrored), -roll_degrees(manager.ref_root_quat), atol=1e-3
    )


def test_sample_root_state_matches_the_table_at_a_known_phase(manager):
    """Phase 0 samples the first frame exactly, for whichever side is paretic."""
    manager.gait_phase[:] = 0.0
    manager.paretic_side[:] = -1.0  # left paretic: the archive's own side
    height, quaternion, linear, angular = manager.sample_root_state()
    assert height[0].item() == pytest.approx(manager.ref_root_height[0].item(), abs=1e-5)
    assert torch.allclose(quaternion[0], manager.ref_root_quat[0], atol=1e-5)
    assert torch.allclose(linear[0], manager.ref_root_lin_vel[0], atol=1e-5)
    assert torch.allclose(angular[0], manager.ref_root_ang_vel[0], atol=1e-5)

    manager.paretic_side[:] = 1.0  # right paretic: the reflection
    _, quaternion, linear, _ = manager.sample_root_state()
    assert torch.allclose(quaternion[0], manager.ref_root_quat_mirrored[0], atol=1e-5)
    assert torch.allclose(linear[0], manager.ref_root_lin_vel_mirrored[0], atol=1e-5)


def test_sample_root_state_interpolates_between_frames(manager):
    """Half a sample apart lands halfway, so the reset is not quantised to 1001 poses."""
    manager.paretic_side[:] = -1.0
    manager.gait_phase[:] = 0.5 / (manager.num_samples - 1)
    height, _, linear, _ = manager.sample_root_state()
    expected_height = 0.5 * (manager.ref_root_height[0] + manager.ref_root_height[1])
    expected_forward = 0.5 * (manager.ref_root_lin_vel[0, 0] + manager.ref_root_lin_vel[1, 0])
    assert height[0].item() == pytest.approx(expected_height.item(), abs=1e-5)
    assert linear[0, 0].item() == pytest.approx(expected_forward.item(), abs=1e-5)


def test_root_height_stays_within_the_measured_band(manager):
    """1.0364 to 1.0664, against the articulation default of 1.05 this replaces."""
    assert manager.ref_root_height.min().item() == pytest.approx(1.0364, abs=0.002)
    assert manager.ref_root_height.max().item() == pytest.approx(1.0664, abs=0.002)


def test_the_gap_that_motivated_this(manager):
    """Quantifies what the old reset was asking of the policy, so a regression is loud.

    Half the reference stride's frames need more than 0.24 m/s of forward momentum that the
    default root state did not supply, while the joints were placed mid-swing as though it
    had. 55.8% of the stride is single support, and that is the regime where the missing
    momentum decides whether the swing foot can be reached or the second foot must be planted.
    """
    forward = manager.ref_root_lin_vel[:, 0]
    assert forward.median().item() > 0.2
    obliquity = roll_degrees(manager.ref_root_quat)
    assert obliquity.min().item() > 1.0, "the reference pelvis is never level"
    assert math.isclose(obliquity.mean().item(), 6.0, abs_tol=1.5)
