# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    # actions
    "ReferenceResidualSpasticAction",
    "ReferenceResidualSpasticActionCfg",
    # observations
    "reference_joint_pos",
    "reference_joint_vel",
    "reference_joint_pos_error",
    "paretic_side",
    "gait_phase",
    # rewards
    "joint_pos_tracking",
    "joint_vel_tracking",
    "track_forward_velocity",
    "track_base_height",
    "compute_xcom_and_mos",
    "margin_of_stability",
    "paretic_foot_clearance",
    "spastic_torque_l2",
    # terminations
    "reference_tracking_divergence",
    # events
    "reset_to_reference_pose",
    "randomize_asymmetric_joint_gains",
    "randomize_asymmetric_effort_limits",
    "randomize_asymmetric_leg_mass",
    "push_biased_toward_paretic_side",
    # curriculums
    "spasticity_ramp",
    "push_magnitude_ramp",
]

# Forward stable MDP terms lazily, then override with environment-specific terms below.
from isaaclab.envs.mdp import *  # noqa: F401, F403

from .actions import ReferenceResidualSpasticAction, ReferenceResidualSpasticActionCfg
from .curriculums import push_magnitude_ramp, spasticity_ramp
from .events import (
    push_biased_toward_paretic_side,
    randomize_asymmetric_effort_limits,
    randomize_asymmetric_joint_gains,
    randomize_asymmetric_leg_mass,
    reset_to_reference_pose,
)
from .observations import (
    gait_phase,
    paretic_side,
    reference_joint_pos,
    reference_joint_pos_error,
    reference_joint_vel,
)
from .rewards import (
    compute_xcom_and_mos,
    joint_pos_tracking,
    joint_vel_tracking,
    margin_of_stability,
    paretic_foot_clearance,
    spastic_torque_l2,
    track_base_height,
    track_forward_velocity,
)
from .terminations import reference_tracking_divergence
