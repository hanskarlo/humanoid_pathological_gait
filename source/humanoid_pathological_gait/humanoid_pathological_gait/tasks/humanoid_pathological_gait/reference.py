# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-environment clinical reference gait state.

Holds the retargeted post-stroke stride and, for every environment, which side is
paretic and where in the stride that environment currently is. Sampling the stride
at an environment's phase gives the joint targets the action term drives toward and
the reward terms score against.

Which of the stride's two legs carries the pathology is read from the archive's
``impaired_side`` field, not assumed. Environments whose paretic side is the other
one read a sagittally mirrored copy, so a single policy learns both presentations.

That field exists because the answer used to be hard-coded here as "left". It was
right for the stride that happened to be at ``--stride-idx 0``, and would have
silently inverted the paretic side -- weakening the leg the reference walks
*normally* on -- for any other stride. An archive written before the field existed
still loads, with the old assumption applied explicitly and a warning.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from .h1_joints import NUM_JOINTS, H1JointLayout


class ReferenceGaitManager:
    """Reference stride playback and per-environment gait phase bookkeeping."""

    def __init__(
        self,
        stride_path: str | Path,
        layout: H1JointLayout,
        num_envs: int,
        device: torch.device | str,
        stride_duration_s: float = 1.2,
    ):
        self.layout = layout
        self.num_envs = num_envs
        self.device = torch.device(device)

        q_clinical, v_clinical, data_duration_s, impaired_side, contact, speed = self._load(Path(stride_path))
        self.stride_duration_s = float(data_duration_s if data_duration_s > 0.0 else stride_duration_s)
        self.impaired_side = impaired_side

        # Store in simulation joint order so every consumer can compare elementwise
        # against ``robot.data.joint_pos`` without another permutation.
        self.ref_q = layout.to_sim_order(q_clinical.to(self.device))
        self.ref_v = layout.to_sim_order(v_clinical.to(self.device))
        self.ref_q_mirrored = layout.mirror(self.ref_q)
        self.ref_v_mirrored = layout.mirror(self.ref_v)
        self.num_samples = self.ref_q.shape[0]

        # ``sample`` hands back the unmirrored stride to left-paretic environments, so the
        # tables are swapped once here if the stride's impaired leg is the right one. Doing
        # it at load time keeps the per-step path free of the branch.
        if self.impaired_side == "right":
            self.ref_q, self.ref_q_mirrored = self.ref_q_mirrored, self.ref_q
            self.ref_v, self.ref_v_mirrored = self.ref_v_mirrored, self.ref_v
            if contact is not None:
                contact = contact[:, ::-1].copy()

        #: Mean forward speed of the stride itself, m/s. The reward tracks this rather than
        #: a fixed constant: the previous 0.5 m/s target was twice what the reference walks
        #: at, and joint tracking outweighs the velocity term 15 to 2, so the two objectives
        #: simply fought. ``nan`` when the archive predates the field.
        self.reference_speed = speed

        #: ``(T, 2)`` contact schedule in **paretic/sound** order, or ``None``. Column 0 is
        #: the impaired limb, matching ``ref_q``'s left slots after the swap above.
        self.ref_contact = (
            torch.tensor(contact, dtype=torch.float32, device=self.device) if contact is not None else None
        )

        # Per-environment state.
        self.gait_phase = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.stride_duration = torch.full((num_envs,), self.stride_duration_s, dtype=torch.float32, device=self.device)
        self.paretic_side = torch.where(
            torch.rand(num_envs, device=self.device) > 0.5,
            torch.ones(num_envs, device=self.device),
            -torch.ones(num_envs, device=self.device),
        )

    def _load(self, path: Path) -> tuple[torch.Tensor, torch.Tensor, float, str, "np.ndarray | None", float]:
        """Load the stride, deriving velocities by finite difference if absent."""
        data = np.load(str(path), allow_pickle=True)
        q = torch.tensor(np.asarray(data["q_trajectory"]), dtype=torch.float32)
        if q.ndim != 2 or q.shape[1] != NUM_JOINTS:
            raise ValueError(f"Reference stride at {path} has shape {tuple(q.shape)}, expected (T, {NUM_JOINTS}).")

        impaired_side = str(data["impaired_side"]) if "impaired_side" in data.files else ""
        if impaired_side not in ("left", "right"):
            # Either an archive predating the field, or a stride too symmetric for the
            # stiff-knee margin to call. Fall back to what this class used to assume, but
            # say so: getting it wrong weakens the leg the reference walks normally on.
            warnings.warn(
                f"Reference stride at {path} declares impaired_side={impaired_side!r}; assuming 'left'. "
                "Regenerate it with data/batch_parse_gait.py to record the side explicitly.",
                RuntimeWarning,
                stacklevel=2,
            )
            impaired_side = "left"

        duration_s = 0.0
        if "time_vector" in data.files:
            time_vector = np.asarray(data["time_vector"])
            duration_s = float(time_vector[-1] - time_vector[0])

        if "v_trajectory" in data.files:
            v = torch.tensor(np.asarray(data["v_trajectory"]), dtype=torch.float32)
        else:
            dt = (duration_s if duration_s > 0.0 else 1.2) / max(q.shape[0] - 1, 1)
            v = torch.gradient(q, spacing=(dt,), dim=0)[0]

        contact = np.asarray(data["reference_contact"]) if "reference_contact" in data.files else None
        if contact is not None and (contact.ndim != 2 or contact.shape != (q.shape[0], 2)):
            raise ValueError(
                f"reference_contact at {path} has shape {contact.shape}, expected {(q.shape[0], 2)}"
            )
        speed = float(data["reference_speed_ms"]) if "reference_speed_ms" in data.files else float("nan")

        return q, v, duration_s, impaired_side, contact, speed

    def advance(self, dt: float, rate_scale: float = 1.0) -> None:
        """Advance every environment's gait phase by ``dt`` seconds of stride time."""
        self.gait_phase = torch.remainder(self.gait_phase + rate_scale * dt / self.stride_duration, 1.0)

    def reset(self, env_ids: torch.Tensor, phase: torch.Tensor | float, paretic_side: torch.Tensor) -> None:
        """Reset gait phase and paretic side for the given environments."""
        self.gait_phase[env_ids] = (
            phase if isinstance(phase, torch.Tensor) else torch.full_like(self.gait_phase[env_ids], float(phase))
        )
        self.paretic_side[env_ids] = paretic_side

    def sample(self, env_ids: torch.Tensor | slice | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate reference joint positions and velocities at the current phases.

        Args:
            env_ids: Environments to sample, or ``None`` for all of them.

        Returns:
            ``(q_ref, v_ref)`` in simulation joint order, shape ``(len(env_ids), 19)``.
        """
        if env_ids is None:
            env_ids = slice(None)
        phase = self.gait_phase[env_ids]
        side = self.paretic_side[env_ids]

        # Linear interpolation between the two bracketing stride samples.
        position = phase * (self.num_samples - 1)
        lower = torch.floor(position).long().clamp_(0, self.num_samples - 1)
        upper = torch.clamp(lower + 1, max=self.num_samples - 1)
        blend = (position - lower.float()).unsqueeze(-1)

        def interpolate(table: torch.Tensor) -> torch.Tensor:
            return torch.lerp(table[lower], table[upper], blend)

        is_right_paretic = (side > 0).unsqueeze(-1)
        q_ref = torch.where(is_right_paretic, interpolate(self.ref_q_mirrored), interpolate(self.ref_q))
        v_ref = torch.where(is_right_paretic, interpolate(self.ref_v_mirrored), interpolate(self.ref_v))
        return q_ref, v_ref

    def sample_contact(self, env_ids: torch.Tensor | slice | None = None) -> torch.Tensor | None:
        """Reference contact state at each environment's phase, as ``(N, 2)`` in ``[0, 1]``.

        Column order is ``(paretic, sound)`` for every environment, matching the frame the
        gait-analysis code standardises to. **The caller must put the robot's own contact
        into that order before comparing**: the robot reports ``(left, right)``, so a
        right-paretic environment needs its two columns swapped.

        Nearest-neighbour rather than interpolated: contact is binary, and a blended value
        midway through a transition would ask the foot to be half-loaded.
        """
        if self.ref_contact is None:
            return None
        if env_ids is None:
            env_ids = slice(None)
        position = self.gait_phase[env_ids] * (self.num_samples - 1)
        index = torch.round(position).long().clamp_(0, self.num_samples - 1)
        return self.ref_contact[index]
