# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Score the reference's own contact schedule with the function that scores every policy.

``single_support_episode_s`` was compared against 1.02 s in three research-log entries, in
the environment config, and in this module's own comment. That number is the reference's
*total* single support per stride -- 51 control steps across two episodes -- and it was being
held up against a policy's *mean episode*. A sum against a mean, which roughly doubled the
apparent gap. The reference's mean episode is 0.5115 s.

Running the reference through the same reduction as the policy is what makes the two sides of
any such comparison commensurable, so every target quoted from the reference is pinned here.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

numpy = pytest.importorskip("numpy")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DATA = (
    ROOT
    / "source/humanoid_pathological_gait/humanoid_pathological_gait/tasks/humanoid_pathological_gait/data"
    / "retargeted_h1_stride.npz"
)

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="reference stride not staged")


@pytest.fixture(scope="module")
def reference_metrics():
    from gait_analysis import contact_gait_metrics

    archive = numpy.load(str(DATA), allow_pickle=True)
    contact = archive["reference_contact"]
    time_vector = archive["time_vector"]
    dt = float(numpy.mean(numpy.diff(time_vector)))

    paretic_column = 0 if str(archive["impaired_side"]) == "left" else 1
    ordered = numpy.stack([contact[:, paretic_column], contact[:, 1 - paretic_column]], axis=-1)
    ordered = ordered[:, None, :]
    valid = numpy.ones(ordered.shape[:2], dtype=bool)
    phase = (numpy.arange(contact.shape[0]) / (contact.shape[0] - 1))[:, None]
    return contact_gait_metrics(ordered, valid, dt, gait_phase=phase)


def test_reference_single_support_episode_is_half_a_second(reference_metrics):
    """The target policies are measured against. 0.5115 s, not the 1.02 s once quoted."""
    assert reference_metrics["single_support_episode_s"] == pytest.approx(0.5115, abs=0.002)


def test_the_total_is_the_number_that_was_being_misquoted(reference_metrics):
    """1.02 s is real, but it is a per-stride total across two episodes, not one episode.

    Pinned so the distinction stays visible: both numbers are legitimate, and which one a
    policy should be compared against depends entirely on which one the policy's number is.
    """
    archive = numpy.load(str(DATA), allow_pickle=True)
    contact = archive["reference_contact"]
    time_vector = archive["time_vector"]
    stride_duration = float(time_vector[-1] - time_vector[0])
    single_share = float(numpy.mean(contact.sum(axis=1) == 1))
    assert single_share * stride_duration == pytest.approx(1.022, abs=0.005)
    assert single_share * stride_duration / reference_metrics["single_support_episode_s"] == pytest.approx(
        2.0, abs=0.02
    ), "the stride contains exactly two single-support episodes"


def test_reference_double_support_matches_the_quoted_schedule(reference_metrics):
    """0.4416, the figure every double-support comparison in this project is drawn against."""
    assert reference_metrics["double_support_fraction"] == pytest.approx(0.4416, abs=0.001)


def test_the_reference_is_not_flagged_as_fragmented(reference_metrics):
    """One contact per foot per cycle. A policy scoring ~4 here is in a different regime."""
    assert reference_metrics["contact_pattern_fragmented"] is False
