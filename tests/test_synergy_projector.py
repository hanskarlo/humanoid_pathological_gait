# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The paretic-leg synergy projector, without Isaac Sim.

Models loss of selective motor control as a rank constraint on the paretic limb's *residual*
-- the corrections it can make -- rather than on its torque magnitude. Clark et al.
(J Neurophysiol 103(2):844-857, 2010) find post-stroke motor modules are merges of the healthy
basis, a rank reduction that predicts walking performance.

Two things are pinned here because both have precedent for going wrong in this project: that
the projector is a genuine orthogonal projector of the requested rank, and that its
right-paretic conjugation is the sagittal reflection rather than a copy. A mirrored
coordination constraint applied to half the environments is the same class of error that
cancelled 85% of the AMP corpus's asymmetry.
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
    return ReferenceGaitManager(str(DATA), layout, num_envs=8, device="cpu"), layout


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
def test_is_an_orthogonal_projector_of_the_requested_rank(manager, rank):
    """P = P^T, P @ P = P, trace(P) = rank. Fails loudly if the SVD is transposed."""
    projector, _ = manager[0].paretic_leg_synergy_projector(rank), None
    assert projector.shape == (5, 5)
    assert torch.allclose(projector, projector.T, atol=1e-5), "not symmetric"
    assert torch.allclose(projector @ projector, projector, atol=1e-5), "not idempotent"
    assert projector.diagonal().sum().item() == pytest.approx(rank, abs=1e-4), "wrong rank"


def test_rank_five_is_the_identity(manager):
    """Full rank must be a no-op, so 5 is a valid 'unimpaired' setting rather than a subtle change."""
    projector = manager[0].paretic_leg_synergy_projector(5)
    assert torch.allclose(projector, torch.eye(5), atol=1e-5)


def test_ranks_are_nested(manager):
    """A lower rank must retain strictly less. Guards against the basis being reordered."""
    reference, _ = manager
    previous = -1.0
    for rank in (1, 2, 3, 4, 5):
        retained = reference.paretic_leg_synergy_projector(rank).diagonal().sum().item()
        assert retained > previous
        previous = retained


def test_rejects_an_out_of_range_rank(manager):
    for bad in (0, 6, -1):
        with pytest.raises(ValueError):
            manager[0].paretic_leg_synergy_projector(bad)


def test_right_paretic_conjugation_is_the_sagittal_reflection(manager):
    """P_right = D P D, and it must be a projector of the same rank -- not a copy of P.

    Under a sagittal reflection hip yaw and hip roll negate while hip pitch, knee and ankle do
    not. Conjugating a symmetric P by that diagonal of +-1 preserves symmetry, idempotence and
    rank, so the reflected constraint is as strong as the original -- but it is a *different*
    subspace, and using P unchanged on right-paretic environments would silently impose a
    mirrored coordination pattern on half the cohort.
    """
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.reference import LEG_SYNERGY_JOINTS

    reference, layout = manager
    projector = reference.paretic_leg_synergy_projector(2)
    columns = torch.tensor([layout.sim_names.index(f"left_{j}") for j in LEG_SYNERGY_JOINTS])
    signs = layout.mirror_sign[columns]

    # The reflection must actually flip something, or the test proves nothing.
    assert (signs < 0).any(), "no leg joint negates under mirroring -- check the layout"

    mirrored = signs.unsqueeze(1) * projector * signs.unsqueeze(0)
    assert torch.allclose(mirrored, mirrored.T, atol=1e-5)
    assert torch.allclose(mirrored @ mirrored, mirrored, atol=1e-5)
    assert mirrored.diagonal().sum().item() == pytest.approx(2.0, abs=1e-4)
    assert not torch.allclose(mirrored, projector, atol=1e-3), "conjugation was a no-op"


def test_projection_removes_out_of_pattern_corrections(manager):
    """The constraint has to bite where the policy actually operates.

    The reference's paretic-leg pattern is sagittal-dominant, and the trained policies correct
    largely in hip yaw and hip roll -- the frontal-plane bracing. A pure hip-roll correction
    must therefore lose most of its magnitude under a rank-2 projection, or the constraint is
    not doing the thing it is being added to do.
    """
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.reference import LEG_SYNERGY_JOINTS

    reference, _ = manager
    projector = reference.paretic_leg_synergy_projector(2)
    hip_roll = torch.zeros(5)
    hip_roll[LEG_SYNERGY_JOINTS.index("hip_roll")] = 1.0
    retained = (hip_roll @ projector).norm().item()
    assert retained < 0.5, f"rank-2 projection retains {retained:.3f} of a pure hip-roll correction"


def test_reference_trajectory_is_already_low_rank(manager):
    """Why the projector is applied to the residual and not to the reference.

    A single periodic stride is intrinsically low-dimensional: this one reaches 90% variance
    explained at rank 2, so constraining the *reference* at any useful rank would change
    almost nothing. Pinned so that anyone tempted to move the projection upstream sees why
    it was not put there.
    """
    reference, _ = manager
    trace = reference.ref_q[:, reference._paretic_leg_columns]
    centred = trace - trace.mean(dim=0, keepdim=True)
    spectrum = torch.linalg.svdvals(centred) ** 2
    cumulative = torch.cumsum(spectrum / spectrum.sum(), dim=0)
    assert cumulative[1].item() > 0.90
