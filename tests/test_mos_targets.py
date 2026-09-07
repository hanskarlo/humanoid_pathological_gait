# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Margin-of-stability targets, checked against the reference's own measured margins.

The old term used a single 0.04 m target -- the reference's *mean*, which it never actually
holds -- with a one-sided formulation that gave full marks to any margin above target and
penalised every negative one. Scored properly the patient's own gait earned ~0.425 per
stride against a shuffling policy's ~0.601, so the reward preferred the shuffle. These tests
pin the property that broke: **the reference must outscore the shuffle.**

Reference margins, measured by MuJoCo FK through the same convention the reward uses:
double support +0.191 +/- 0.024 over 44.2% of the cycle, single support -0.082 +/- 0.029
over the remaining 55.8%.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

#: Measured reference margins per support state, metres.
REF_DOUBLE_MEAN, REF_DOUBLE_STD = 0.1906, 0.0235
REF_SINGLE_MEAN, REF_SINGLE_STD = -0.0818, 0.0287
REF_DOUBLE_SHARE = 0.442

#: What a shuffling policy actually held (spastic_knee, model_1200).
SHUFFLE_MARGIN = 0.1701


EXT = pathlib.Path(__file__).resolve().parents[1] / "source" / "humanoid_pathological_gait"
sys.path.insert(0, str(EXT))
CFG = (
    EXT
    / "humanoid_pathological_gait/tasks/humanoid_pathological_gait/config/h1_pathological"
    / "h1_pathological_env_cfg.py"
)


@pytest.fixture(scope="module")
def params():
    """The reward's own defaults.

    Read by introspection rather than from the task config: importing the config pulls in
    ``isaaclab.envs``, which bootstraps Isaac Kit and cannot run under pytest.
    ``test_config_does_not_override_the_targets`` covers the gap.
    """
    import inspect

    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.mdp import rewards

    signature = inspect.signature(rewards.margin_of_stability)
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def test_config_does_not_override_the_targets(params):
    """The task config must pass the same margins these tests are checking."""
    text = CFG.read_text()
    for key in ("double_support_margin", "single_support_margin"):
        assert f'"{key}": {params[key]}' in text, f"config disagrees with the default for {key}"


def _score(mos, params):
    """The reward's tracking term, for a margin measured in the given support state."""
    mos = np.asarray(mos, dtype=float)
    double = np.asarray(mos) > 0.05  # regime implied by the margin's own sign/size
    target = np.where(double, params["double_support_margin"], params["single_support_margin"])
    return np.exp(-((mos - target) ** 2) / params["std"] ** 2)


def test_targets_match_the_measured_regimes(params):
    """Each target must sit within a standard deviation of the regime it represents."""
    assert abs(params["double_support_margin"] - REF_DOUBLE_MEAN) < REF_DOUBLE_STD
    assert abs(params["single_support_margin"] - REF_SINGLE_MEAN) < REF_SINGLE_STD


def test_the_regimes_are_actually_distinguished(params):
    """A single target cannot serve both: they must be far apart relative to ``std``."""
    separation = params["double_support_margin"] - params["single_support_margin"]
    assert separation > 4 * params["std"], (
        f"targets {separation:.3f} m apart with std {params['std']} would blur the two "
        "regimes back into one"
    )


def test_single_support_target_is_negative(params):
    """The physiological point: real single support runs the XCoM outside the support foot."""
    assert params["single_support_margin"] < 0.0


def test_the_reference_outscores_the_shuffle(params):
    """The property whose absence caused the shuffle.

    A stride alternating between the reference's two measured regimes must score strictly
    better than holding a constant over-large margin. Under the old one-sided term this was
    false -- 0.425 against 0.601 -- which is exactly a reward that forbids single support.
    """
    reference = REF_DOUBLE_SHARE * _score(REF_DOUBLE_MEAN, params) + (1.0 - REF_DOUBLE_SHARE) * _score(
        REF_SINGLE_MEAN, params
    )
    shuffle = _score(SHUFFLE_MARGIN, params)

    assert reference > shuffle, f"reference scores {reference:.3f} against shuffle {shuffle:.3f}"
    assert reference > 0.9, f"the reference should nearly max this term, got {reference:.3f}"


def test_over_stability_is_no_longer_free(params):
    """Two-sided: exceeding the double-support target must cost something.

    The old term clamped the shortfall at zero, so every margin above target scored 1.0 and
    a policy sitting at 0.17 had no gradient pulling it down.
    """
    at_target = _score(params["double_support_margin"], params)
    well_above = _score(params["double_support_margin"] + 4 * params["std"], params)

    assert at_target > 0.99
    assert well_above < 0.2, "margins far above target still score full marks"


def test_reference_extremes_stay_credited(params):
    """The reference's own worst margins must not be treated as failures."""
    for extreme in (-0.1411, -0.0391, 0.1524, 0.2330):
        assert _score(extreme, params) > 0.05, f"reference margin {extreme} scores near zero"
