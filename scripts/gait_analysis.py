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


def pelvic_obliquity(root_quat, is_right_paretic, valid, contact_schedule=None):
    """Coronal pelvic obliquity in degrees, and the hiking signature if a schedule is given.

    Pelvic hiking is one of the hemiparetic hallmarks this project set out to reproduce, and it
    was claimed without ever being measured. It is also the mechanism the *reference* uses for
    lateral foot clearance: obliquity rises from +4.25 deg in paretic stance to +8.75 in
    paretic swing, a **+4.50 deg hiking signature**, while the paretic hip's entire range of
    motion is 5.47 deg and never leaves adduction. Circumduction in this data is pelvic, not
    femoral -- so a claim that it emerges from hip adductor spasticity is not supported.

    Sign is standardised so positive means hiked on the paretic side: obliquity negates under a
    left/right reflection, so right-paretic environments are flipped before averaging. Without
    that step a balanced cohort cancels the very asymmetry being measured, which is the same
    trap the knee symmetry index and the AMP corpus both fell into.

    Returns ``(mean_deg, hiking_signature_deg)``; the signature is ``nan`` without a schedule.
    """
    # Isaac Lab quaternions are (x, y, z, w) in this release.
    x, y, z, w = root_quat[..., 0], root_quat[..., 1], root_quat[..., 2], root_quat[..., 3]
    roll = np.degrees(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    roll = np.where(np.asarray(is_right_paretic)[None, :], -roll, roll)

    mean_deg = float(np.mean(roll[valid])) if valid.any() else float("nan")
    signature = float("nan")
    if contact_schedule is not None:
        swing = contact_schedule & valid
        stance = (~contact_schedule) & valid
        if swing.any() and stance.any():
            signature = float(np.mean(roll[swing]) - np.mean(roll[stance]))
    return mean_deg, signature


def contact_gait_metrics(
    contact: np.ndarray, valid: np.ndarray, dt: float, gait_phase: np.ndarray | None = None
) -> dict[str, float]:
    """Stance/swing timing and temporal asymmetry from a per-foot contact mask.

    Args:
        contact: ``(steps, num_envs, 2)`` boolean, ordered ``(paretic, sound)``.
        valid: ``(steps, num_envs)`` boolean; steps in an episode that never terminated.
        dt: Control timestep in seconds.
        gait_phase: ``(steps, num_envs)`` in ``[0, 1)``, used only to count gait cycles so
            that stance periods per cycle can be reported. Optional; without it the
            fragmentation guard below cannot fire.

    Returns:
        Stance fraction and mean single-support step time per limb, plus the temporal
        symmetry index, **signed** as ``(Tp - Ts) / (Tp + Ts) * 100``: 0% for a symmetric
        gait and *negative* for the shortened paretic stance that characterizes hemiparetic
        walking. The sign is the clinically meaningful part and the unsigned form cannot
        distinguish hemiparesis from its mirror image, so both are returned.

        ``double_support_fraction`` is returned alongside because the asymmetry indices say
        nothing about whether the robot is stepping at all.
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
    # Signed, paretic minus sound. Hemiparetic gait shortens paretic stance, so a genuinely
    # hemiparetic policy is NEGATIVE here. The unsigned form this replaces could not tell
    # hemiparesis from its mirror image: the 2026-09-05 baseline scored a clinically
    # plausible 16.4% on a seed whose paretic limb bore weight *longer* than its sound one.
    signed = 100.0 * (paretic_time - sound_time) / denominator if np.isfinite(denominator) else float("nan")

    # Stance time above is the mean duration of a contact run, which equals the clinical
    # stance time only when each foot makes exactly ONE contact per gait cycle. A shuffling
    # policy breaks that: the 2026-09-05 baseline made 2.3-2.7 contacts per cycle on two of
    # three seeds, and there the run-duration and stance-fraction definitions of asymmetry
    # came out with OPPOSITE signs. Counting the runs is what tells a reader which regime
    # they are in, so the asymmetry is withheld rather than reported misleadingly.
    periods_per_cycle = [float("nan"), float("nan")]
    if gait_phase is not None:
        cycles = float(np.sum(np.diff(gait_phase, axis=0) < -0.5))
        if cycles > 0:
            for foot in range(contact.shape[2]):
                runs = int(np.sum(np.diff(masked[:, :, foot].astype(np.int8), axis=0) == 1))
                periods_per_cycle[foot] = runs / cycles

    fragmented = any(np.isfinite(v) and not (0.6 <= v <= 1.6) for v in periods_per_cycle)
    if fragmented:
        signed = float("nan")

    # Mean duration of one uninterrupted single-support episode, in seconds. This separates
    # two behaviours the double-support *fraction* conflates: a policy that stands on one leg
    # for a real step, and one that unloads briefly and often. Measured, they differ sharply
    # -- 4.2 control steps per episode against the reference stride's ~51 -- and a fading
    # balance assist moved the fraction by nine standard deviations while making the episodes
    # *shorter*. Report this alongside the fraction or that distinction is invisible.
    # ``masked`` zeroes invalid samples, which would read as "no foot loaded" and split an
    # episode in two, so the validity mask is applied to the episode test as well.
    single = (contact.sum(axis=-1) == 1) & valid
    single_support_runs: list[float] = []
    for env_index in range(single.shape[1]):
        edges = np.diff(single[:, env_index].astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        # Same treatment as the stance runs above: an episode clipped by the recording
        # window is incomplete and would bias the mean downward.
        if starts.size and ends.size:
            ends = ends[ends > starts[0]]
            single_support_runs.extend((ends - starts[: ends.size]) * dt)
    single_support_s = float(np.mean(single_support_runs)) if single_support_runs else float("nan")

    return {
        "single_support_episode_s": single_support_s,
        "paretic_stance_periods_per_cycle": periods_per_cycle[0],
        "sound_stance_periods_per_cycle": periods_per_cycle[1],
        # True when the contact pattern is not one-stance-per-cycle, which makes the
        # clinical stance-time definition inapplicable and blanks the asymmetry above.
        "contact_pattern_fragmented": bool(fragmented),
        "paretic_stance_fraction": float(stance_fraction[0]),
        "sound_stance_fraction": float(stance_fraction[1]),
        "paretic_stance_time_s": paretic_time,
        "sound_stance_time_s": sound_time,
        # Negative = paretic stance shorter = the hemiparetic direction.
        "temporal_asymmetry_pct": float(signed),
        # Magnitude only, for comparison against literature that reports it unsigned.
        "temporal_asymmetry_magnitude_pct": float(abs(signed)),
        # Fraction of the cycle with both feet loaded. Normal walking is 0.20-0.25; this
        # cohort's own patients sit at 0.37; a shuffling policy runs far higher, and none
        # of the other metrics here reveal that on their own.
        # Restricted to `valid` like every other metric here; counting post-termination
        # steps would fold a collapsed robot's two grounded feet into the walking statistic.
        "double_support_fraction": float(np.mean(contact.all(axis=-1)[valid])) if valid.any() else float("nan"),
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
