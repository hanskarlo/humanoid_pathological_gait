# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The swing-load constraint's math, on schedules and loads where the answer is known.

A termination probability that fires in stance, misses the core of a swing, or ignores which foot
is which would train against the wrong thing and look plausible doing it. These pin the three
properties that matter: only core swing counts, the worse foot decides, and the CaT mapping is
``p_max * clip((load - epsilon) / c_max, 0, 1)``.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

EXT = pathlib.Path(__file__).resolve().parents[1] / "source" / "humanoid_pathological_gait"
sys.path.insert(0, str(EXT))


@pytest.fixture(scope="module")
def sc():
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait import swing_constraint

    return swing_constraint


def schedule(n: int = 100, swing: tuple[int, int] = (40, 70)) -> torch.Tensor:
    """One foot down except in ``[swing[0], swing[1])``; the other foot always down."""
    contact = torch.ones(n, 2, dtype=torch.bool)
    contact[swing[0] : swing[1], 0] = False
    return contact


def test_core_swing_drops_the_margin_at_both_ends(sc) -> None:
    core = sc.core_swing_mask(schedule(), margin=5)
    assert core[:, 0].nonzero().flatten().tolist() == list(range(45, 65))
    assert not core[:, 1].any(), "a foot that is always down has no swing"


def test_core_swing_wraps_across_the_stride_boundary(sc) -> None:
    contact = torch.ones(100, 2, dtype=torch.bool)
    contact[90:, 0] = False
    contact[:10, 0] = False  # swing 90..109, i.e. across the end of the array
    core = sc.core_swing_mask(contact, margin=3)
    assert core[:, 0].nonzero().flatten().tolist() == [*range(0, 7), *range(93, 100)]


def test_zero_margin_is_the_whole_swing(sc) -> None:
    contact = schedule()
    assert torch.equal(sc.core_swing_mask(contact, margin=0), ~contact)


def test_probability_follows_the_cat_mapping(sc) -> None:
    in_swing = torch.tensor([[True, False]] * 4)
    load = torch.tensor([[0.04, 0.0], [0.05, 0.0], [0.175, 0.0], [0.9, 0.0]])
    p = sc.violation_probability(load, in_swing, p_max=0.1, epsilon=0.05, c_max=0.25)
    assert torch.allclose(p, torch.tensor([0.0, 0.0, 0.05, 0.1]))


def test_load_outside_core_swing_never_terminates(sc) -> None:
    load = torch.tensor([[1.0, 1.0]])
    p = sc.violation_probability(load, torch.tensor([[False, False]]), p_max=0.25)
    assert p.item() == 0.0


def test_the_worse_foot_decides(sc) -> None:
    load = torch.tensor([[0.10, 0.30]])
    p = sc.violation_probability(load, torch.tensor([[True, True]]), p_max=0.2, epsilon=0.05, c_max=0.25)
    assert torch.isclose(p, torch.tensor([0.2])).all()


def test_zero_p_max_is_inert(sc) -> None:
    load = torch.tensor([[1.0, 1.0]])
    assert sc.violation_probability(load, torch.tensor([[True, True]]), p_max=0.0).item() == 0.0


def test_core_stance_is_the_complement_trimmed_at_both_ends(sc) -> None:
    core = sc.core_stance_mask(schedule(), margin=5)
    # foot 0 is down outside [40, 70): its core stance is [75, 100) and [0, 35), cyclically
    assert core[:, 0].nonzero().flatten().tolist() == [*range(0, 35), *range(75, 100)]
    assert core[:, 1].all(), "a foot that is always down is always in core stance"


def test_unload_probability_follows_its_mapping(sc) -> None:
    in_stance = torch.tensor([[True, False]] * 4)
    load = torch.tensor([[0.0, 0.0], [0.025, 0.0], [0.05, 0.0], [0.5, 0.0]])
    p = sc.unload_probability(load, in_stance, p_max=0.1, epsilon=0.05)
    assert torch.allclose(p, torch.tensor([0.1, 0.05, 0.0, 0.0]))


def test_unloading_outside_core_stance_never_terminates(sc) -> None:
    p = sc.unload_probability(torch.tensor([[0.0, 0.0]]), torch.tensor([[False, False]]), p_max=0.25)
    assert p.item() == 0.0
