# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint layout for the 19-DoF Unitree H1 used by the pathological-gait task.

Two orderings are in play and conflating them is the easiest way to silently
corrupt this task:

* **Clinical order** -- the order the retargeted stride in ``data/`` is stored in
  (:data:`CLINICAL_JOINT_ORDER`): left leg, right leg, torso, left arm, right arm.
* **Simulation order** -- whatever order the H1 articulation reports from
  ``Articulation.joint_names``, which interleaves the two sides.

Everything downstream of :class:`~.reference.ReferenceGaitManager` works in
simulation order, so it can be compared elementwise against
``robot.data.joint_pos``. :class:`H1JointLayout` owns the one conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

CLINICAL_JOINT_ORDER: tuple[str, ...] = (
    # left leg
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    # right leg
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    # torso
    "torso",
    # left arm
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    # right arm
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
)
"""Joint order of the retargeted reference stride stored under ``data/``."""

NUM_JOINTS = len(CLINICAL_JOINT_ORDER)

LEFT_LEG_JOINTS: tuple[str, ...] = CLINICAL_JOINT_ORDER[0:5]
RIGHT_LEG_JOINTS: tuple[str, ...] = CLINICAL_JOINT_ORDER[5:10]
ARM_JOINTS: tuple[str, ...] = CLINICAL_JOINT_ORDER[11:19]

FOOT_BODY_NAMES: tuple[str, str] = ("left_ankle_link", "right_ankle_link")
"""Bodies whose contact state defines the support polygon (H1 has no separate foot link)."""

#: Joints whose sign flips when the gait is mirrored to the other side. Pitch-axis
#: joints (hip pitch, knee, ankle, shoulder pitch, elbow) are symmetric about the
#: sagittal plane and keep their sign; yaw- and roll-axis joints reverse.
_MIRROR_ANTISYMMETRIC_SUFFIXES: tuple[str, ...] = ("_hip_yaw", "_hip_roll", "_shoulder_roll", "_shoulder_yaw")

#: Per-joint clinical tracking weight. Ankles carry the most weight (foot drop is the
#: headline deficit), then knees (stiff-knee / hyperextension), then the rest of the leg.
#: Arms matter least -- their reference is a static posture, not a measured signal.
CLINICAL_TRACKING_WEIGHTS: dict[str, float] = {
    "hip_yaw": 1.0,
    "hip_roll": 1.5,
    "hip_pitch": 2.0,
    "knee": 2.5,
    "ankle": 3.0,
    "torso": 1.0,
    "shoulder_pitch": 0.5,
    "shoulder_roll": 0.5,
    "shoulder_yaw": 0.5,
    "elbow": 0.5,
}


def _mirror_name(name: str) -> str:
    """Return the contralateral joint name (``torso`` mirrors onto itself)."""
    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    return name


def _mirror_sign(name: str) -> float:
    """Return ``-1`` for joints whose value negates under a left/right mirror."""
    if name == "torso":
        return -1.0
    return -1.0 if any(name.endswith(suffix) for suffix in _MIRROR_ANTISYMMETRIC_SUFFIXES) else 1.0


def _tracking_weight(name: str) -> float:
    """Return the clinical tracking weight for a joint name."""
    for suffix, weight in CLINICAL_TRACKING_WEIGHTS.items():
        if name.endswith(suffix):
            return weight
    raise KeyError(f"No clinical tracking weight defined for joint '{name}'.")


@dataclass
class H1JointLayout:
    """Index bookkeeping between the clinical and simulation joint orderings.

    Attributes:
        sim_names: Joint names in simulation order.
        clinical_to_sim: Gather index that reorders a clinical-order tensor into
            simulation order; see :meth:`to_sim_order`.
        sim_to_clinical: The inverse gather index; see :meth:`to_clinical_order`.
        mirror_index: Gather index (simulation order) mapping each joint onto its
            contralateral partner.
        mirror_sign: Per-joint sign applied alongside :attr:`mirror_index`.
        tracking_weights: Per-joint clinical tracking weights in simulation order.
        left_leg_ids / right_leg_ids / arm_ids: Simulation-order joint indices.
    """

    sim_names: tuple[str, ...]
    clinical_to_sim: torch.Tensor
    sim_to_clinical: torch.Tensor
    mirror_index: torch.Tensor
    mirror_sign: torch.Tensor
    tracking_weights: torch.Tensor
    left_leg_ids: torch.Tensor
    right_leg_ids: torch.Tensor
    arm_ids: torch.Tensor

    @classmethod
    def from_sim_names(cls, sim_names: list[str], device: torch.device | str) -> H1JointLayout:
        """Build the layout from an articulation's reported joint names.

        Raises:
            ValueError: If the articulation's joints are not exactly the 19 expected ones.
        """
        if set(sim_names) != set(CLINICAL_JOINT_ORDER):
            missing = sorted(set(CLINICAL_JOINT_ORDER) - set(sim_names))
            extra = sorted(set(sim_names) - set(CLINICAL_JOINT_ORDER))
            raise ValueError(
                "H1 articulation joints do not match the expected 19-DoF pathological-gait layout. "
                f"Missing: {missing}. Unexpected: {extra}."
            )

        sim_index_of = {name: i for i, name in enumerate(sim_names)}
        # Gather index that turns a clinical-order tensor into a simulation-order one:
        # entry i holds the clinical slot that supplies simulation joint i.
        clinical_to_sim = torch.tensor(
            [CLINICAL_JOINT_ORDER.index(name) for name in sim_names], dtype=torch.long, device=device
        )
        sim_to_clinical = torch.empty_like(clinical_to_sim)
        sim_to_clinical[clinical_to_sim] = torch.arange(len(sim_names), dtype=torch.long, device=device)
        mirror_index = torch.tensor(
            [sim_index_of[_mirror_name(name)] for name in sim_names], dtype=torch.long, device=device
        )
        mirror_sign = torch.tensor([_mirror_sign(name) for name in sim_names], dtype=torch.float32, device=device)
        tracking_weights = torch.tensor(
            [_tracking_weight(name) for name in sim_names], dtype=torch.float32, device=device
        )

        def ids(names: tuple[str, ...]) -> torch.Tensor:
            return torch.tensor([sim_index_of[n] for n in names], dtype=torch.long, device=device)

        return cls(
            sim_names=tuple(sim_names),
            clinical_to_sim=clinical_to_sim,
            sim_to_clinical=sim_to_clinical,
            mirror_index=mirror_index,
            mirror_sign=mirror_sign,
            tracking_weights=tracking_weights,
            left_leg_ids=ids(LEFT_LEG_JOINTS),
            right_leg_ids=ids(RIGHT_LEG_JOINTS),
            arm_ids=ids(ARM_JOINTS),
        )

    def to_sim_order(self, clinical: torch.Tensor) -> torch.Tensor:
        """Reorder a ``(..., 19)`` clinical-order tensor into simulation order."""
        return clinical[..., self.clinical_to_sim]

    def to_clinical_order(self, sim: torch.Tensor) -> torch.Tensor:
        """Reorder a ``(..., 19)`` simulation-order tensor into clinical order."""
        return sim[..., self.sim_to_clinical]

    def mirror(self, values: torch.Tensor) -> torch.Tensor:
        """Mirror a ``(..., 19)`` simulation-order tensor across the sagittal plane."""
        return self.mirror_sign * values[..., self.mirror_index]

    def index_of(self, name: str) -> int:
        """Simulation-order index of a named joint."""
        return self.sim_names.index(name)
