# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The swing-load constraint: no weight on a foot the reference has in the air.

Forward plan 2026-10-13, Phase 2. Every arm fails single support. In subject 0 the specific
violation is the paretic foot coming down in its long scheduled swing: it is loaded on 34-69 % of
those steps at 0.5-0.75 body weight, and that breaks the sound limb's 0.72 s single support
(``research_log/2026-10-14-correction-the-reference-contact-was-swapped-twice.md``). Eight reward
interventions, ``swing_timing`` among them, never removed it, because a reward term is outbid.

So this is a *constraint*, in the form of Constraints as Terminations (Chane-Sane et al., IROS
2024). A violation ends the episode with a probability that grows with the violation, so the
attractor is made unreachable rather than merely expensive. The probability is ramped in from zero
by a curriculum, because applied at full strength to a policy with no single-support skill it would
end every episode (2026-09-07).

Pure torch, no Isaac Lab, so the two functions below are testable on synthetic data
(``tests/test_swing_constraint.py``).
"""

from __future__ import annotations

import torch


def core_swing_mask(contact: torch.Tensor, margin: int) -> torch.Tensor:
    """Scheduled swing, minus ``margin`` samples at each end, treating the stride as a cycle.

    Args:
        contact: ``(samples, 2)`` boolean reference contact schedule, True where the foot is down.
        margin: Samples dropped at each end of every swing, so that toe-off and touch-down are not
            judged to the sample. The constraint is about mid-swing support, not timing.

    Returns:
        ``(samples, 2)`` boolean, True where the foot is in its swing and at least ``margin``
        samples from either end of it.
    """
    swing = ~contact.bool()
    core = swing.clone()
    for shift in range(1, margin + 1):
        core &= torch.roll(swing, shifts=shift, dims=0) & torch.roll(swing, shifts=-shift, dims=0)
    return core


def violation_probability(
    load_fraction: torch.Tensor,
    in_core_swing: torch.Tensor,
    p_max: float,
    epsilon: float = 0.05,
    c_max: float = 0.25,
) -> torch.Tensor:
    """Per-environment termination probability, as in CaT: ``p_max * clip(violation / c_max, 0, 1)``.

    Args:
        load_fraction: ``(N, 2)`` foot load as a fraction of body weight, (paretic, sound).
        in_core_swing: ``(N, 2)`` boolean, the foot's reference phase is inside its core swing.
        p_max: Termination probability at full violation, per control step (the curriculum's
            current value).
        epsilon: Load tolerated in swing, as a fraction of body weight (a graze is not support).
        c_max: Violation above ``epsilon`` at which the probability reaches ``p_max``.

    Returns:
        ``(N,)`` probabilities in ``[0, p_max]``; the worse foot decides.
    """
    violation = torch.clamp(load_fraction - epsilon, min=0.0) * in_core_swing.float()
    return p_max * torch.clamp(violation / c_max, 0.0, 1.0).amax(dim=-1)
