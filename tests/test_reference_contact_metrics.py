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


def test_reference_stance_asymmetry_is_negative(reference_metrics):
    """-15.87%: the paretic limb bears weight for a smaller fraction of the cycle.

    This is the defining temporal signature of hemiparetic gait and the sign is the whole
    content of the number. It is pinned because every policy trained in this project scores
    it *positive* -- the mirror image of the pathology -- and nothing was reporting it: the
    run-duration index that was supposed to catch this is blanked by the fragmentation guard
    on every policy run, and the fraction-based index did not exist.
    """
    assert reference_metrics["stance_fraction_asymmetry_pct"] == pytest.approx(-15.87, abs=0.05)


def test_stance_asymmetry_survives_fragmentation():
    """A fraction does not care how the loaded time is divided, which is why this metric exists.

    Chops the reference's two stance periods into many short ones without changing how long
    either foot is loaded in total. The run-duration index gives up; this one is unmoved.
    """
    from gait_analysis import contact_gait_metrics

    archive = numpy.load(str(DATA), allow_pickle=True)
    contact = archive["reference_contact"].copy()
    dt = float(numpy.mean(numpy.diff(archive["time_vector"])))

    # Three strides end to end, so the gait phase actually wraps and the fragmentation guard
    # has cycles to count -- it cannot fire on a single non-wrapping cycle.
    strides = 3
    tiled = numpy.tile(contact, (strides, 1))
    # Punch a one-sample hole every 20 samples: same total loaded time, shredded runs.
    tiled[::20] = False
    ordered = numpy.stack([tiled[:, 0], tiled[:, 1]], axis=-1)[:, None, :]
    valid = numpy.ones(ordered.shape[:2], dtype=bool)
    within = numpy.arange(contact.shape[0]) / contact.shape[0]
    phase = numpy.tile(within, strides)[:, None]
    metrics = contact_gait_metrics(ordered, valid, dt, gait_phase=phase)

    assert metrics["contact_pattern_fragmented"] is True
    assert numpy.isnan(metrics["temporal_asymmetry_pct"]), "the run-duration index gives up"
    assert metrics["stance_fraction_asymmetry_pct"] == pytest.approx(-15.87, abs=0.5)


def test_reference_patterson_symmetry_ratio(reference_metrics):
    """Patterson et al. 2008's published ratio, computed on the reference: 3.29.

    SR = (paretic swing/stance) / (nonparetic swing/stance); 1.0 is symmetric and the
    hemiparetic direction is above 1. Pinned separately from the project's own asymmetry
    indices because it is the only one of the three that is comparable to published values,
    and only for an unfragmented gait.
    """
    assert reference_metrics["patterson_symmetry_ratio"] == pytest.approx(3.29, abs=0.02)


def test_patterson_ratio_and_stance_asymmetry_agree_in_direction(reference_metrics):
    """Two definitions, one pathology: SR above 1 must coincide with a negative fraction index.

    They are different functions of the same two stance fractions, so a sign disagreement
    would mean one of them is implemented backwards -- which is exactly the class of error
    that let an inverted weight-bearing asymmetry go unnoticed for eighteen runs.
    """
    assert reference_metrics["patterson_symmetry_ratio"] > 1.0
    assert reference_metrics["stance_fraction_asymmetry_pct"] < 0.0
