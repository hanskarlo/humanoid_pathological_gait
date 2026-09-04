# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms that fade the pathology in over training.

Spasticity and reduced paretic torque make the task considerably harder than plain
reference tracking, and switching them on at full strength from iteration zero tends
to leave the policy stuck on the floor. Both curricula start the deficit low and ramp
it linearly to its configured value, so the policy learns the reference stride first
and then learns to fight the pathology.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.managers import CurriculumTermCfg

    from ..h1_pathological_env import H1PathologicalGaitEnv


def _linear_ramp(step: int, start_step: int, num_steps: int, start_value: float, end_value: float) -> float:
    """Value of a linear ramp from ``start_value`` to ``end_value`` at ``step``."""
    if num_steps <= 0:
        return end_value
    progress = min(max((step - start_step) / num_steps, 0.0), 1.0)
    return start_value + progress * (end_value - start_value)


def spasticity_ramp(
    env: H1PathologicalGaitEnv,
    env_ids: torch.Tensor,
    start_step: int = 0,
    num_steps: int = 24_000,
    start_scale: float = 0.0,
    end_scale: float = 1.0,
) -> float:
    """Ramp the TSRT reflex gain from ``start_scale`` to ``end_scale``.

    Returns the current scale so the curriculum manager logs it.
    """
    scale = _linear_ramp(env.common_step_counter, start_step, num_steps, start_scale, end_scale)
    env.spasticity_scale.fill_(scale)
    return scale


def push_magnitude_ramp(
    env: H1PathologicalGaitEnv,
    env_ids: torch.Tensor,
    term_name: str = "push_robot",
    start_step: int = 0,
    num_steps: int = 24_000,
    start_scale: float = 0.0,
    end_scale: float = 1.0,
) -> float:
    """Ramp the magnitude of the balance-perturbation pushes.

    Reads the term's configured ranges once and rescales them each call, so the
    configured values stay the ramp's endpoint rather than drifting.
    """
    term_cfg: CurriculumTermCfg = env.event_manager.get_term_cfg(term_name)
    if term_name not in env.base_push_velocity_range:
        env.base_push_velocity_range[term_name] = {
            axis: tuple(bounds) for axis, bounds in term_cfg.params["velocity_range"].items()
        }

    scale = _linear_ramp(env.common_step_counter, start_step, num_steps, start_scale, end_scale)
    term_cfg.params["velocity_range"] = {
        axis: (low * scale, high * scale) for axis, (low, high) in env.base_push_velocity_range[term_name].items()
    }
    env.event_manager.set_term_cfg(term_name, term_cfg)
    return scale
