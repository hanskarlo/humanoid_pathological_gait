# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-specific termination terms.

Falls and time-outs are covered by stock Isaac Lab terms. What an imitation task
additionally wants is to stop an episode once the robot has diverged far enough
from the reference that the remaining rollout carries no useful learning signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..h1_pathological_env import H1PathologicalGaitEnv


def reference_tracking_divergence(
    env: H1PathologicalGaitEnv,
    max_rms_error: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the clinically weighted RMS joint tracking error exceeds a bound.

    Args:
        env: The environment.
        max_rms_error: Weighted RMS joint position error (rad) at which to give up.
        asset_cfg: The articulation being tracked.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q_ref, _ = env.reference_gait.sample()
    weights = env.joint_layout.tracking_weights
    weighted_mse = torch.sum(torch.square(asset.data.joint_pos.torch - q_ref) * weights, dim=-1) / torch.sum(weights)
    return torch.sqrt(weighted_mse) > max_rms_error


def swing_load_constraint(
    env: H1PathologicalGaitEnv,
    sensor_cfg: SceneEntityCfg,
    epsilon: float = 0.05,
    c_max: float = 0.25,
    margin_s: float = 0.04,
) -> torch.Tensor:
    """Constraint as termination: end the episode, with a probability set by the violation, when a
    foot carries weight in the core of its scheduled swing.

    See :mod:`..swing_constraint` for why this is a constraint and not another reward term. The
    probability is ``env.swing_constraint_p_max * clip((load / body weight - epsilon) / c_max, 0, 1)``,
    the worse foot deciding. ``env.swing_constraint_p_max`` is set by the ``swing_constraint_ramp``
    curriculum and is 0 until it runs.

    Training only. ``evaluate.py`` does not restore it: the trained policy is judged without the
    constraint, as in CaT.

    Args:
        env: The environment.
        sensor_cfg: The contact sensor and the two foot bodies, in (left, right) order.
        epsilon: Load tolerated in swing, as a fraction of body weight.
        c_max: Violation above ``epsilon`` at which the probability reaches ``p_max``.
        margin_s: Seconds dropped at each end of every scheduled swing, so that toe-off and
            touch-down are not judged to the sample.
    """
    from ..swing_constraint import core_swing_mask, violation_probability

    reference = env.reference_gait
    if reference.ref_contact is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    cache = getattr(env, "_swing_constraint_cache", None)
    if cache is None:
        sample_dt = float(reference.stride_duration_s) / (reference.num_samples - 1)
        core = core_swing_mask(reference.ref_contact > 0.5, int(round(margin_s / sample_dt)))
        # Nominal weight, before the mass randomisation. The threshold is a fraction of the body
        # the policy was built for, not of each draw.
        body_weight = env.default_body_mass.to(env.device).sum(dim=-1) * 9.81
        cache = env._swing_constraint_cache = (core, body_weight)
    core, body_weight = cache

    index = torch.round(reference.gait_phase * (reference.num_samples - 1)).long().clamp_(0, reference.num_samples - 1)
    in_core_swing = core[index]  # (N, 2), (paretic, sound) like the schedule itself

    sensor = env.scene[sensor_cfg.name]
    force = torch.norm(sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids], dim=-1)
    # The sensor reports (left, right); the schedule is (paretic, sound).
    is_right_paretic = (reference.paretic_side > 0).unsqueeze(-1)
    force = torch.where(is_right_paretic, force.flip(-1), force)

    probability = violation_probability(
        force / body_weight.unsqueeze(-1),
        in_core_swing,
        float(getattr(env, "swing_constraint_p_max", 0.0)),
        epsilon,
        c_max,
    )
    return torch.rand_like(probability) < probability


def stance_unload_constraint(
    env: H1PathologicalGaitEnv,
    sensor_cfg: SceneEntityCfg,
    epsilon: float = 0.05,
    margin_s: float = 0.04,
) -> torch.Tensor:
    """Constraint as termination: a foot (nearly) unloaded in the core of its scheduled stance.

    The companion of :func:`swing_load_constraint`; see :mod:`..swing_constraint`. The probability
    is ``env.stance_constraint_p_max * clip((epsilon - load / body weight) / epsilon, 0, 1)``, the
    worse foot deciding. It is set by the ``stance_constraint_ramp`` curriculum, and is 0 until
    that runs. Training only; ``evaluate.py`` does not restore it.
    """
    from ..swing_constraint import core_stance_mask, unload_probability

    reference = env.reference_gait
    if reference.ref_contact is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    cache = getattr(env, "_stance_constraint_cache", None)
    if cache is None:
        sample_dt = float(reference.stride_duration_s) / (reference.num_samples - 1)
        core = core_stance_mask(reference.ref_contact > 0.5, int(round(margin_s / sample_dt)))
        body_weight = env.default_body_mass.to(env.device).sum(dim=-1) * 9.81
        cache = env._stance_constraint_cache = (core, body_weight)
    core, body_weight = cache

    index = torch.round(reference.gait_phase * (reference.num_samples - 1)).long().clamp_(0, reference.num_samples - 1)
    in_core_stance = core[index]  # (N, 2), (paretic, sound)

    sensor = env.scene[sensor_cfg.name]
    force = torch.norm(sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids], dim=-1)
    is_right_paretic = (reference.paretic_side > 0).unsqueeze(-1)
    force = torch.where(is_right_paretic, force.flip(-1), force)

    probability = unload_probability(
        force / body_weight.unsqueeze(-1), in_core_stance, float(getattr(env, "stance_constraint_p_max", 0.0)), epsilon
    )
    return torch.rand_like(probability) < probability
