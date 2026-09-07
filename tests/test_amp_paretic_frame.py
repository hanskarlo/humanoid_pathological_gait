# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP features must reach the discriminator in one canonical paretic frame.

The discriminator sees raw left/right joint slots with no indication of which side is
impaired. A corpus mixing left- and right-paretic patients therefore presents it with a
distribution that is symmetric in aggregate: measured on this corpus the knee ROM difference
visible to the discriminator was 1.68 deg against the 10.95 deg the pathology carries, so
the style reward -- the largest single weight in the task at 5.0 -- was asking for a nearly
symmetric gait. Canonicalising both sides of the comparison restores it to 13.51 deg.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

EXT = pathlib.Path(__file__).resolve().parents[1] / "source" / "humanoid_pathological_gait"
sys.path.insert(0, str(EXT))
CORPUS = EXT / "humanoid_pathological_gait/tasks/humanoid_pathological_gait/data/amp_expert_corpus.npz"


@pytest.fixture(scope="module")
def layout():
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.h1_joints import (
        CLINICAL_JOINT_ORDER,
        H1JointLayout,
    )

    return H1JointLayout.from_sim_names(list(CLINICAL_JOINT_ORDER), device="cpu")


def _features(batch=4):
    from humanoid_pathological_gait.algorithms.amp.discriminator import AMP_FEATURE_DIM

    torch.manual_seed(0)
    return torch.randn(batch, AMP_FEATURE_DIM)


def test_mirroring_twice_is_the_identity(layout):
    from humanoid_pathological_gait.algorithms.amp import mirror_amp_features

    features = _features()
    once = mirror_amp_features(features, layout.mirror_index, layout.mirror_sign)
    twice = mirror_amp_features(once, layout.mirror_index, layout.mirror_sign)

    torch.testing.assert_close(twice, features)
    assert not torch.allclose(once, features), "mirroring changed nothing at all"


def test_angular_velocity_is_treated_as_a_pseudovector(layout):
    """Roll and yaw negate under the reflection; pitch does not. Linear y negates."""
    from humanoid_pathological_gait.algorithms.amp import mirror_amp_features

    features = _features(1)
    out = mirror_amp_features(features, layout.mirror_index, layout.mirror_sign)

    assert out[0, 2] == -features[0, 2]  # projected gravity y
    assert out[0, 5] == -features[0, 5]  # linear velocity y
    assert out[0, 7] == -features[0, 7]  # angular velocity x (roll)
    assert out[0, 8] == features[0, 8]  # angular velocity y (pitch) -- unchanged
    assert out[0, 9] == -features[0, 9]  # angular velocity z (yaw)
    assert out[0, 0] == features[0, 0]  # root height


def test_to_paretic_frame_only_mirrors_right_paretic(layout):
    from humanoid_pathological_gait.algorithms.amp import mirror_amp_features, to_paretic_frame

    features = _features(4)
    is_right = torch.tensor([False, True, False, True])
    out = to_paretic_frame(features, is_right, layout.mirror_index, layout.mirror_sign)
    mirrored = mirror_amp_features(features, layout.mirror_index, layout.mirror_sign)

    torch.testing.assert_close(out[0], features[0])
    torch.testing.assert_close(out[2], features[2])
    torch.testing.assert_close(out[1], mirrored[1])
    torch.testing.assert_close(out[3], mirrored[3])


@pytest.mark.skipif(not CORPUS.exists(), reason="expert corpus not staged")
def test_corpus_presents_real_asymmetry_to_the_discriminator():
    """The property that was broken: a paretic knee that is visibly stiffer than the sound one."""
    from humanoid_pathological_gait.algorithms.amp import AMPExpertMotionBuffer
    from humanoid_pathological_gait.algorithms.amp.discriminator import AMP_JOINT_POS
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.h1_joints import CLINICAL_JOINT_ORDER

    buffer = AMPExpertMotionBuffer(dataset_path=str(CORPUS), device="cpu")
    names = list(CLINICAL_JOINT_ORDER)
    left, right = names.index("left_knee"), names.index("right_knee")

    rom = {"left": [], "right": []}
    for trajectory in buffer.trajectories:
        joints = trajectory[:, AMP_JOINT_POS].numpy()
        rom["left"].append(np.degrees(np.ptp(joints[:, left])))
        rom["right"].append(np.degrees(np.ptp(joints[:, right])))

    paretic, sound = np.mean(rom["left"]), np.mean(rom["right"])
    assert paretic < sound, (
        f"canonical-paretic knee ROM {paretic:.2f} deg is not stiffer than the sound side "
        f"{sound:.2f} deg -- the corpus is mirrored the wrong way, or is using the inverted "
        "'paretic_sides' label field instead of the motion-inferred 'impaired_sides'"
    )
    assert sound - paretic > 8.0, (
        f"only {sound - paretic:.2f} deg of asymmetry reaches the discriminator; mixing the "
        "two paretic sides cancels it to about 1.7 deg"
    )


@pytest.mark.skipif(not CORPUS.exists(), reason="expert corpus not staged")
def test_the_inverted_label_field_is_not_used():
    """Guard the P/N inversion at corpus level.

    ``paretic_sides`` comes from the dataset's own labels and disagrees with the
    motion-inferred ``impaired_sides`` on 84.5% of strides. Only the inferred field puts the
    pathology the right way round.
    """
    data = np.load(CORPUS, allow_pickle=True)
    assert "impaired_sides" in data, "corpus predates motion-inferred sides; rebuild it"

    inferred = data["impaired_sides"].astype(str)
    labelled = np.asarray(data["paretic_sides"])
    agreement = np.mean((inferred == "left") == (labelled == 0))

    assert agreement < 0.5, (
        f"the two side fields now agree on {agreement:.1%} of strides. They disagreed on "
        "84.5% when the inversion was characterised, so either the corpus was rebuilt or "
        "the fields changed meaning -- re-derive which one is trustworthy before relying on it."
    )
