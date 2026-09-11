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


LEG_SYNERGY_JOINTS: tuple[str, ...] = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
"""Leg joints spanned by the synergy projector, in the order its basis rows are expressed in."""


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
            root_quaternion,
            reference_mos,
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

        #: ``(19,)`` per-joint reference range of motion in radians, and its mirror. See
        #: :meth:`joint_rom` for why the tracking reward needs this.
        #: Simulation-order columns of the paretic leg, in LEG_SYNERGY_JOINTS order. The
        #: archive's own impaired side is the left in the canonical (unmirrored) tables.
        self._paretic_leg_columns = torch.tensor(
            [layout.sim_names.index(f"left_{joint}") for joint in LEG_SYNERGY_JOINTS],
            dtype=torch.long,
            device=self.device,
        )

        self.ref_joint_rom = self.ref_q.max(dim=0).values - self.ref_q.min(dim=0).values
        self.ref_joint_rom_mirrored = (
            self.ref_q_mirrored.max(dim=0).values - self.ref_q_mirrored.min(dim=0).values
        )

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

        #: ``(T, 4)`` reference pelvis orientation in simulation order ``(x, y, z, w)``, with
        #: the stride's own heading rotation removed so it composes with a randomised yaw.
        #:
        #: The reference pelvis is **not** level. It carries +1.78 to +9.65 deg of coronal
        #: roll -- the pelvic obliquity that produces this stride's lateral foot clearance,
        #: +4.25 deg in paretic stance rising to +8.75 in paretic swing. Resetting the root
        #: to a level default discards the very hallmark the policy is asked to reproduce.
        self.ref_root_quat = None

        #: ``(T, 3)`` reference pelvis linear velocity in the stride's heading frame, m/s.
        #:
        #: Mean forward +0.2444, ranging +0.0917 to +0.4351. Placing the joints on the
        #: reference while leaving the root at rest is not a noisy version of the reference
        #: state, it is a physically inconsistent one: the legs are mid-swing and the body
        #: has no momentum to swing over. See :meth:`sample_root_state`.
        self.ref_root_lin_vel = None

        #: ``(T, 3)`` reference pelvis angular velocity in the stride's heading frame, rad/s.
        self.ref_root_ang_vel = None

        #: Sagittal reflections of the three tables above, for right-paretic environments.
        self.ref_root_quat_mirrored = None
        self.ref_root_lin_vel_mirrored = None
        self.ref_root_ang_vel_mirrored = None

        #: ``(T,)`` mediolateral margin of stability of the reference, in metres, or ``None``.
        #:
        #: Not swapped for a right-paretic stride, unlike ``ref_contact``. The margin is
        #: ``min(upper_edge - xcom, xcom - lower_edge)``; under a left/right reflection the
        #: two terms exchange places and the minimum is unchanged, so MoS is mirror-invariant
        #: and the trajectory over phase is the same either way.
        self.ref_mos = (
            torch.tensor(reference_mos, dtype=torch.float32, device=self.device)
            if reference_mos is not None
            else None
        )

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
            self._build_root_state_tables(root_translation, root_quaternion, angle)

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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        float,
        str,
        "np.ndarray | None",
        float,
        "np.ndarray | None",
        "np.ndarray | None",
        "np.ndarray | None",
    ]:
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

        root_quaternion = (
            np.asarray(data["root_quaternion"]) if "root_quaternion" in data.files else None
        )
        if root_quaternion is not None and root_quaternion.shape != (q.shape[0], 4):
            raise ValueError(
                f"root_quaternion at {path} has shape {root_quaternion.shape}, expected {(q.shape[0], 4)}"
            )

        reference_mos = np.asarray(data["reference_mos"]) if "reference_mos" in data.files else None
        if reference_mos is not None and reference_mos.shape[0] != q.shape[0]:
            raise ValueError(
                f"reference_mos at {path} has {reference_mos.shape[0]} frames, expected {q.shape[0]}"
            )

        return (
            q,
            v,
            duration_s,
            impaired_side,
            contact,
            speed,
            root_translation,
            root_quaternion,
            reference_mos,
        )

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

    def sample_mos_target(self, env_ids: torch.Tensor | slice | None = None) -> torch.Tensor | None:
        """Reference mediolateral margin of stability at each environment's phase, in metres.

        Written into the stride archive by ``scripts/add_reference_mos.py``; ``None`` when the
        archive predates it, which lets the reward fall back to its two per-regime constants.

        Those constants -- +0.19 in double support, -0.08 in single -- capture the regime
        split but not the variation inside it: the reference's within-regime spread is
        0.024 and 0.029 m against the reward's own 0.05 m width, so half the stride scored
        between 0.5 and 0.9 against its own target. Per-phase tracking is what took pelvis
        height from the same problem to a clean 1.0.
        """
        if self.ref_mos is None:
            return None
        if env_ids is None:
            env_ids = slice(None)
        position = self.gait_phase[env_ids] * (self.num_samples - 1)
        lower = torch.floor(position).long().clamp_(0, self.num_samples - 1)
        upper = (lower + 1).clamp_(max=self.num_samples - 1)
        alpha = (position - lower.float()).clamp_(0.0, 1.0)
        return torch.lerp(self.ref_mos[lower], self.ref_mos[upper], alpha)

    def _build_root_state_tables(self, root_translation, root_quaternion, angle: torch.Tensor) -> None:
        """Derive heading-frame root orientation and velocity tables from the stride archive.

        Everything here exists so a reset can place the *whole* root state on the reference,
        not just its height. The archive stores the pelvis pose in the capture's world frame;
        an episode starts at an arbitrary heading, so the stride's own heading rotation is
        removed once here and a random yaw is composed back on at reset time. That keeps this
        consistent with ``ref_root_disp``, which is rotated by the same ``angle``.

        Quaternion conventions are the trap in this function. The archive is MuJoCo-ordered
        ``(w, x, y, z)``; this Isaac Lab release is ``(x, y, z, w)`` in both the data buffers
        and ``isaaclab.utils.math``. The stored table is in *simulation* order. The check that
        the ordering is right is that the coronal roll recovered from it reproduces the
        reference's measured pelvic obliquity -- +4.25 deg in paretic stance, +8.75 in paretic
        swing -- which ``tests/test_reference_root_state.py`` asserts.
        """
        if root_quaternion is None:
            return

        numpy_dtype = np.float64
        quaternion = np.asarray(root_quaternion, dtype=numpy_dtype)
        translation = np.asarray(root_translation, dtype=numpy_dtype)
        num_samples = quaternion.shape[0]
        dt = self.stride_duration_s / max(num_samples - 1, 1)

        # A quaternion and its negation are the same rotation, and the archive is free to
        # flip sign between frames. Differentiating across a flip would fabricate an angular
        # velocity spike of 2/dt, so make the sequence continuous first.
        flip = np.sign(np.sum(quaternion[1:] * quaternion[:-1], axis=1))
        flip[flip == 0.0] = 1.0
        quaternion[1:] *= np.cumprod(flip)[:, None]

        def multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            """Hamilton product of ``(w, x, y, z)`` quaternions, broadcasting over frames."""
            w1, x1, y1, z1 = first[..., 0], first[..., 1], first[..., 2], first[..., 3]
            w2, x2, y2, z2 = second[..., 0], second[..., 1], second[..., 2], second[..., 3]
            return np.stack(
                [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ],
                axis=-1,
            )

        heading_angle = float(angle)
        planar_cos, planar_sin = np.cos(-heading_angle), np.sin(-heading_angle)
        unrotate = np.array([[planar_cos, -planar_sin], [planar_sin, planar_cos]], dtype=numpy_dtype)

        # Pelvis orientation with the stride's heading removed.
        yaw_inverse = np.array(
            [np.cos(-heading_angle / 2.0), 0.0, 0.0, np.sin(-heading_angle / 2.0)], dtype=numpy_dtype
        )
        heading_frame_quat = multiply(np.broadcast_to(yaw_inverse, quaternion.shape), quaternion)

        # Angular velocity from the quaternion derivative: omega = 2 * (dq/dt) * conj(q).
        quaternion_rate = np.gradient(heading_frame_quat, dt, axis=0)
        conjugate = heading_frame_quat * np.array([1.0, -1.0, -1.0, -1.0], dtype=numpy_dtype)
        angular_velocity = 2.0 * multiply(quaternion_rate, conjugate)[:, 1:]

        linear_velocity = np.gradient(translation, dt, axis=0)
        linear_velocity[:, :2] = linear_velocity[:, :2] @ unrotate.T

        def to_tensor(values: np.ndarray) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.float32, device=self.device)

        # Store in simulation order (x, y, z, w).
        self.ref_root_quat = to_tensor(heading_frame_quat[:, [1, 2, 3, 0]])
        self.ref_root_lin_vel = to_tensor(linear_velocity)
        self.ref_root_ang_vel = to_tensor(angular_velocity)

        # A right-paretic stride is the sagittal reflection of this one, and the reflection
        # acts differently on each quantity. Position and linear velocity are ordinary
        # vectors, so only the lateral component negates. Angular velocity is a pseudovector:
        # roll and yaw negate, pitch does not -- the same rule the AMP feature mirror uses.
        # A quaternion's vector part follows the pseudovector rule, its scalar part is
        # invariant. Getting any one of these wrong reverses the pelvic obliquity on half the
        # environments, which is exactly how the AMP corpus cancelled 85% of its own signal.
        self.ref_root_quat_mirrored = self.ref_root_quat * torch.tensor(
            [-1.0, 1.0, -1.0, 1.0], device=self.device
        )
        self.ref_root_lin_vel_mirrored = self.ref_root_lin_vel * torch.tensor(
            [1.0, -1.0, 1.0], device=self.device
        )
        self.ref_root_ang_vel_mirrored = self.ref_root_ang_vel * torch.tensor(
            [-1.0, 1.0, -1.0], device=self.device
        )

    def paretic_leg_synergy_projector(self, rank: int) -> torch.Tensor:
        """``(5, 5)`` orthogonal projector onto the paretic leg's top-``rank`` coordination modes.

        Built by PCA on the reference stride's own paretic-leg joint trajectory, over
        ``(hip_yaw, hip_roll, hip_pitch, knee, ankle)`` in that order. Returned in the
        **left-paretic frame**; the caller conjugates it by the mirror signs for right-paretic
        environments.

        This is the loss of selective motor control, expressed as a constraint on what
        *corrections* the paretic limb can make: it may move within the coordination pattern
        its own gait already uses, and not outside it. That is Fugl-Meyer's "moving within
        versus outside synergy" and the merged-module finding of Clark et al.
        (J Neurophysiol 103(2):844-857, 2010), where paretic modules are merges of the healthy
        basis rather than a new one -- a rank reduction.

        **It is an analogue, not the same object, and the difference matters.** Clark's modules
        are NMF factors of EMG across eight muscles with independent activation timing; this is
        PCA over five joint angles. Ranks do not transfer between the two, which is why the
        rank here is chosen from a measured effect on this robot rather than from 3.6-versus-2.7.

        The projector is applied to the *residual*, never to the reference itself. Projecting
        the reference would constrain nothing: a single periodic stride is intrinsically
        low-dimensional and this one is already rank 2 at 90% variance explained, so a
        reference-space constraint at any useful rank is vacuous. The residual is not --
        measured on the best current policy, only 15% of paretic residual energy lies inside
        the rank-2 subspace and 36% inside rank 4, *less* than a random subspace of the same
        rank would retain. The policy's corrections are actively anti-aligned with the
        reference's coordination pattern, concentrated in hip yaw and hip roll: the
        frontal-plane bracing. Constraining the residual forbids exactly that.
        """
        if not 1 <= rank <= len(LEG_SYNERGY_JOINTS):
            raise ValueError(f"synergy rank must be in 1..{len(LEG_SYNERGY_JOINTS)}, got {rank}")
        trace = self.ref_q[:, self._paretic_leg_columns]
        centred = trace - trace.mean(dim=0, keepdim=True)
        # Right singular vectors are the coordination directions; rows of Vh, largest first.
        basis = torch.linalg.svd(centred, full_matrices=False).Vh[:rank]
        return basis.T @ basis

    def joint_rom(self, env_ids: torch.Tensor | slice | None = None) -> torch.Tensor:
        """Per-joint reference range of motion in radians, ``(N, 19)``, in the paretic frame.

        Phase-independent -- it is a property of the stride, not of where in the stride an
        environment is -- but it is still per-environment, because a right-paretic stride is
        the mirror of a left-paretic one and the two put the small ROMs on opposite sides.

        Exists so ``joint_pos_tracking`` can size its RBF width per joint. With one shared
        width of 0.35 rad the term could not resolve motion smaller than itself, and 16 of
        this stride's 19 joints have a total ROM below that: a policy that froze every joint
        at its reference mean scored **0.9106** of the term carrying the largest weight in
        the reward. The paretic limb is where the small ROMs are, so it was the limb the
        objective could least see.
        """
        if env_ids is None:
            env_ids = slice(None)
        is_right_paretic = (self.paretic_side[env_ids] > 0).unsqueeze(-1)
        return torch.where(is_right_paretic, self.ref_joint_rom_mirrored, self.ref_joint_rom)

    def sample_root_state(
        self, env_ids: torch.Tensor | slice | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Full reference root state at each environment's phase, in the stride heading frame.

        Returns ``(height, quaternion, linear_velocity, angular_velocity)`` with shapes
        ``(N,)``, ``(N, 4)`` in ``(x, y, z, w)``, ``(N, 3)`` and ``(N, 3)``; or ``None`` when
        the stride archive predates ``root_quaternion``, so a reset can fall back to the
        default root pose rather than fail.

        Sagittally reflected for right-paretic environments, like :meth:`sample`.
        """
        if self.ref_root_quat is None or self.ref_root_height is None:
            return None
        if env_ids is None:
            env_ids = slice(None)

        position = self.gait_phase[env_ids] * (self.num_samples - 1)
        lower = torch.floor(position).long().clamp_(0, self.num_samples - 1)
        upper = (lower + 1).clamp_(max=self.num_samples - 1)
        alpha = (position - lower.float()).clamp_(0.0, 1.0)

        def interpolate(table: torch.Tensor) -> torch.Tensor:
            blend = alpha if table.ndim == 1 else alpha.unsqueeze(-1)
            return torch.lerp(table[lower], table[upper], blend)

        is_right_paretic = (self.paretic_side[env_ids] > 0).unsqueeze(-1)
        # Normalised lerp rather than slerp: adjacent frames of a 1001-sample stride are
        # under 0.05 deg apart, where the two agree to far better than the 0.02 rad of reset
        # noise applied on top.
        quaternion = torch.where(
            is_right_paretic,
            interpolate(self.ref_root_quat_mirrored),
            interpolate(self.ref_root_quat),
        )
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        linear_velocity = torch.where(
            is_right_paretic,
            interpolate(self.ref_root_lin_vel_mirrored),
            interpolate(self.ref_root_lin_vel),
        )
        angular_velocity = torch.where(
            is_right_paretic,
            interpolate(self.ref_root_ang_vel_mirrored),
            interpolate(self.ref_root_ang_vel),
        )
        return interpolate(self.ref_root_height), quaternion, linear_velocity, angular_velocity

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
