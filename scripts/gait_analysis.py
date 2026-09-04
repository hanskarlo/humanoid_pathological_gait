# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gait analysis on a recorded rollout: pure NumPy, no simulator.

Split out of ``evaluate.py`` so the reductions that produce every number in the paper can
be exercised without booting Isaac Sim -- both by the unit tests and by anyone re-deriving
a metric from an archived ``rollout.npz`` months after the run.
"""

from __future__ import annotations

import numpy as np

NUM_CYCLE_BINS = 101
"""0-100% of the gait cycle inclusive: the convention clinical gait figures are drawn in."""


def standardize_to_paretic_frame(values: np.ndarray, is_right_paretic: np.ndarray, mirror_index, mirror_sign):
    """Mirror right-paretic environments so the paretic limb always sits in the left slots.

    Args:
        values: ``(steps, num_envs, 19)`` in simulation joint order.
        is_right_paretic: ``(num_envs,)`` boolean mask.
        mirror_index: Simulation-order gather index onto the contralateral joint.
        mirror_sign: Per-joint sign flip that accompanies the gather.

    Returns:
        The same array with every right-paretic environment reflected, so that averaging
        across environments preserves rather than cancels the paretic/sound asymmetry.
    """
    mirrored = values[..., mirror_index] * mirror_sign
    return np.where(is_right_paretic[None, :, None], mirrored, values)


def contact_gait_metrics(contact: np.ndarray, valid: np.ndarray, dt: float) -> dict[str, float]:
    """Stance/swing timing and temporal asymmetry from a per-foot contact mask.

    Args:
        contact: ``(steps, num_envs, 2)`` boolean, ordered ``(paretic, sound)``.
        valid: ``(steps, num_envs)`` boolean; steps in an episode that never terminated.
        dt: Control timestep in seconds.

    Returns:
        Stance fraction and mean single-support step time per limb, plus the temporal
        symmetry index. The index is the standard ``|Tp - Ts| / (Tp + Ts) * 100``: 0% for
        a symmetric gait, and rising with the shortened paretic stance that characterizes
        hemiparetic walking.
    """
    valid_steps = valid.sum()
    if valid_steps == 0:
        return {"temporal_asymmetry_pct": float("nan")}

    masked = contact & valid[..., None]
    stance_fraction = masked.sum(axis=(0, 1)) / valid_steps

    # Contact runs: a rising edge starts a stance period, a falling edge ends it.
    step_times = []
    for foot in range(contact.shape[2]):
        durations = []
        for env_index in range(contact.shape[1]):
            trace = masked[:, env_index, foot].astype(np.int8)
            edges = np.diff(trace)
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)
            # Keep only complete stance periods; a run clipped by the recording window
            # would bias the mean downward.
            if starts.size and ends.size:
                ends = ends[ends > starts[0]]
                starts = starts[: ends.size]
                durations.extend((ends - starts) * dt)
        step_times.append(float(np.mean(durations)) if durations else float("nan"))

    paretic_time, sound_time = step_times
    denominator = paretic_time + sound_time
    asymmetry = 100.0 * abs(paretic_time - sound_time) / denominator if np.isfinite(denominator) else float("nan")

    return {
        "paretic_stance_fraction": float(stance_fraction[0]),
        "sound_stance_fraction": float(stance_fraction[1]),
        "paretic_stance_time_s": paretic_time,
        "sound_stance_time_s": sound_time,
        "temporal_asymmetry_pct": float(asymmetry),
    }


def cycle_normalize(values: np.ndarray, phase: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bin a signal onto 0-100% of the gait cycle.

    Clinical gait curves are plotted against cycle percentage, not time, because that is
    what makes strides of different duration comparable and what lets a simulated curve be
    overlaid on a normative band. Binning by the environment's own gait phase also keeps
    the result meaningful when environments reset at different times, which a raw time
    series does not.

    Args:
        values: ``(steps, num_envs, D)``.
        phase: ``(steps, num_envs)`` in ``[0, 1)``.
        valid: ``(steps, num_envs)`` boolean.

    Returns:
        ``(mean, std)``, each ``(NUM_CYCLE_BINS, D)``, with empty bins as NaN.
    """
    num_features = values.shape[-1]
    bins = np.clip((phase * (NUM_CYCLE_BINS - 1)).round().astype(int), 0, NUM_CYCLE_BINS - 1)

    mean = np.full((NUM_CYCLE_BINS, num_features), np.nan)
    std = np.full((NUM_CYCLE_BINS, num_features), np.nan)
    flat_bins = bins[valid]
    flat_values = values[valid]
    for index in range(NUM_CYCLE_BINS):
        selected = flat_values[flat_bins == index]
        if selected.size:
            mean[index] = selected.mean(axis=0)
            std[index] = selected.std(axis=0)
    return mean, std
