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

        (
            q_clinical,
            v_clinical,
            data_duration_s,
            impaired_side,
            contact,
            speed,
            root_translation,
        ) = self._load(Path(stride_path))
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

        #: ``(2,)`` fraction of the cycle each limb is meant to be **down**, paretic first.
        #: The reward uses it to cancel the class imbalance: the schedule says "down" far
        #: more often than "up", so unweighted agreement pays a policy that simply never
        #: lifts a foot. See ``mdp.rewards.swing_timing``.
        self.contact_stance_fraction = (
            self.ref_contact.mean(dim=0) if self.ref_contact is not None else None
        )

        #: ``(T, 2)`` planar displacement of the reference root from the start of the
        #: stride, in the stride's own heading frame. This is what pins *cadence and stride
        #: length*: the joint reference alone says which pose to hold at each phase, which a
        #: policy can satisfy while taking three short steps per cycle instead of one long
        #: one -- measured at 3.58 stance periods per cycle and a 0.185 m stride against the
        #: reference's 0.447 m. Requiring the root to be where the reference's root is, at
        #: the phase the reference is at, states that constraint directly.
        self.ref_root_disp = None

        #: ``(T,)`` absolute pelvis height of the reference, in metres, or ``None``.
        #:
        #: Kept because the pelvis is *not* at a constant height: it rises and falls 30.0 mm
        #: peak to peak over the stride (mean 1.0523, min 1.0364, max 1.0664). The height
        #: reward used to track a constant 1.05, which asks the policy to hold still
        #: vertically while every other term asks it to walk, and caps the achievable score
        #: at whatever the reference's own bob costs. Narrowing that constant target's ``std``
        #: to charge harder for crouching was measured and made every gait metric worse --
        #: see research_log/2026-09-07. The defect was the target, not the width.
        self.ref_root_height = None

        if root_translation is not None:
            planar = torch.tensor(root_translation[:, :2], dtype=torch.float32, device=self.device)
            heading = planar[-1] - planar[0]
            angle = torch.atan2(heading[1], heading[0])
            cos, sin = torch.cos(-angle), torch.sin(-angle)
            rotation = torch.tensor([[cos, -sin], [sin, cos]], device=self.device)
            self.ref_root_disp = (planar - planar[0]) @ rotation.T
            self.ref_root_height = torch.tensor(
                root_translation[:, 2], dtype=torch.float32, device=self.device
            )

        #: Per-environment anchor: root position and heading at the last phase wrap. The
        #: reference displacement is measured from the start of *its* stride, so the robot's
        #: has to be measured from the start of the cycle it is currently in.
        self.cycle_anchor_pos = torch.zeros(num_envs, 2, dtype=torch.float32, device=self.device)
        self.cycle_anchor_yaw = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        #: Gait phase at which each anchor was set. The reference's displacement is measured
        #: from phase 0, but an anchor is not always laid at phase 0: resets randomise the
        #: start phase in training, and ``evaluate.py`` deliberately spreads phases across
        #: environments. Subtracting the reference displacement *at the anchor's phase* is
        #: what makes the error mean "progress since the anchor" rather than "distance from
        #: where a stride that began at phase 0 would be" -- which charged an environment
        #: resetting at phase 0.75 with 0.34 m of debt it could not repay.
        self.cycle_anchor_phase = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        #: Set by :meth:`advance` for the environments whose phase wrapped this step.
        self.cycle_wrapped = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        # Per-environment state.
        self.gait_phase = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.stride_duration = torch.full((num_envs,), self.stride_duration_s, dtype=torch.float32, device=self.device)
        self.paretic_side = torch.where(
            torch.rand(num_envs, device=self.device) > 0.5,
            torch.ones(num_envs, device=self.device),
            -torch.ones(num_envs, device=self.device),
        )

    def _load(
        self, path: Path
    ) -> tuple[torch.Tensor, torch.Tensor, float, str, "np.ndarray | None", float, "np.ndarray | None"]:
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

        root_translation = (
            np.asarray(data["root_translation"]) if "root_translation" in data.files else None
        )
        if root_translation is not None and root_translation.shape[0] != q.shape[0]:
            raise ValueError(
                f"root_translation at {path} has {root_translation.shape[0]} frames, expected {q.shape[0]}"
            )

        return q, v, duration_s, impaired_side, contact, speed, root_translation

    def advance(self, dt: float, rate_scale: float = 1.0) -> None:
        """Advance every environment's gait phase by ``dt`` seconds of stride time.

        Also flags the environments whose phase wrapped, so the caller can re-anchor their
        root displacement: the reference's displacement restarts at zero each stride, and
        the robot's has to restart with it.
        """
        previous = self.gait_phase
        self.gait_phase = torch.remainder(previous + rate_scale * dt / self.stride_duration, 1.0)
        self.cycle_wrapped = self.gait_phase < previous

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

    def set_cycle_anchor(self, position_xy: torch.Tensor, yaw: torch.Tensor, env_ids=None) -> None:
        """Re-anchor the root-displacement origin, recording the phase it was laid at."""
        if env_ids is None:
            env_ids = slice(None)
        self.cycle_anchor_pos[env_ids] = position_xy
        self.cycle_anchor_yaw[env_ids] = yaw
        self.cycle_anchor_phase[env_ids] = self.gait_phase[env_ids]

    def root_progression_error(self, position_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor | None:
        """``(N, 2)`` how far the root is from where the reference's root would be.

        Both displacements are expressed in the heading the environment had when its cycle
        began, so the error is ``(along-track, cross-track)`` in metres and does not depend
        on which way the robot happens to be facing now.
        """
        if self.ref_root_disp is None:
            return None
        def displacement_at(phase: torch.Tensor) -> torch.Tensor:
            index = torch.round(phase * (self.num_samples - 1)).long().clamp_(0, self.num_samples - 1)
            return self.ref_root_disp[index]

        # Progress the reference makes between the anchor's phase and the current one. The
        # anchor's own displacement has to come off, or an environment that started
        # mid-stride is charged for the part of the stride it never ran.
        target = displacement_at(self.gait_phase) - displacement_at(self.cycle_anchor_phase)

        delta = position_xy - self.cycle_anchor_pos
        cos, sin = torch.cos(-self.cycle_anchor_yaw), torch.sin(-self.cycle_anchor_yaw)
        actual = torch.stack(
            [cos * delta[:, 0] - sin * delta[:, 1], sin * delta[:, 0] + cos * delta[:, 1]], dim=-1
        )
        return actual - target

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

    def sample_root_height(self, env_ids: torch.Tensor | slice | None = None) -> torch.Tensor | None:
        """Reference pelvis height at each environment's phase, as ``(N,)`` in metres.

        Linearly interpolated, unlike :meth:`sample_contact`: height is continuous, and
        rounding to the nearest of 1001 samples would put a 30 um staircase on a signal
        whose whole amplitude is 30 mm.

        Returns ``None`` when the stride archive predates ``root_translation``, which lets
        the reward fall back to its constant target rather than fail.
        """
        if self.ref_root_height is None:
            return None
        if env_ids is None:
            env_ids = slice(None)
        position = self.gait_phase[env_ids] * (self.num_samples - 1)
        lower = torch.floor(position).long().clamp_(0, self.num_samples - 1)
        upper = (lower + 1).clamp_(max=self.num_samples - 1)
        alpha = (position - lower.float()).clamp_(0.0, 1.0)
        return torch.lerp(self.ref_root_height[lower], self.ref_root_height[upper], alpha)
