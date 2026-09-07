#!/usr/bin/env python3
"""
discriminator.py

Asymmetric Adversarial Motion Prior (AMP) Discriminator and Reference Motion Buffer.
Provides:
  - Asymmetric Discriminator Network D_psi(s_amp, s'_amp)
  - Least-Squares GAN (LSGAN) Objective with Gradient Penalty
  - Style Reward formulation: r_amp = max(0, 1 - 0.25*(D(s, s') - 1)^2)
  - Replay buffer and Expert Motion Dataset loader for clinical stroke gait trajectories.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ==============================================================================
# 1. Asymmetric AMP Motion State Feature Formulator
# ==============================================================================

# AMP feature vector dimension:
# Root height (1) + Projected Gravity (3) + Root LinVel (3) + Root AngVel (3) + Joint Pos (19) + Joint Vel (19)
# Dimension = 1 + 3 + 3 + 3 + 19 + 19 = 48
AMP_FEATURE_DIM = 48
AMP_TRANSITION_DIM = AMP_FEATURE_DIM * 2  # (s_t, s_{t+1}) -> 96

#: Where each block sits in the 48-dimensional feature vector.
AMP_GRAVITY_Y = 2
AMP_LIN_VEL_Y = 5
AMP_ANG_VEL_X, AMP_ANG_VEL_Z = 7, 9
AMP_JOINT_POS = slice(10, 29)
AMP_JOINT_VEL = slice(29, 48)


def mirror_amp_features(
    features: torch.Tensor, mirror_index: torch.Tensor, mirror_sign: torch.Tensor
) -> torch.Tensor:
    """Reflect AMP features through the sagittal plane, swapping left and right.

    Linear quantities negate their ``y`` component. Angular velocity is a pseudovector, so
    under the same reflection it is ``x`` and ``z`` that negate and ``y`` (pitch) that does
    not -- getting this backwards would mirror the pose while leaving the turn rate telling
    the discriminator which way the robot really leaned.
    """
    mirrored = features.clone()
    mirrored[..., AMP_GRAVITY_Y] = -features[..., AMP_GRAVITY_Y]
    mirrored[..., AMP_LIN_VEL_Y] = -features[..., AMP_LIN_VEL_Y]
    mirrored[..., AMP_ANG_VEL_X] = -features[..., AMP_ANG_VEL_X]
    mirrored[..., AMP_ANG_VEL_Z] = -features[..., AMP_ANG_VEL_Z]
    mirrored[..., AMP_JOINT_POS] = mirror_sign * features[..., AMP_JOINT_POS][..., mirror_index]
    mirrored[..., AMP_JOINT_VEL] = mirror_sign * features[..., AMP_JOINT_VEL][..., mirror_index]
    return mirrored


def to_paretic_frame(
    features: torch.Tensor,
    is_right_paretic: torch.Tensor,
    mirror_index: torch.Tensor,
    mirror_sign: torch.Tensor,
) -> torch.Tensor:
    """Express features in a canonical **left-paretic** frame.

    The discriminator sees raw left/right joint slots and no indication of which side is
    impaired, so a corpus mixing left- and right-paretic patients presents it with a
    distribution that is symmetric in aggregate. Measured on this corpus: the knee range-of-
    motion difference visible to the discriminator was **1.68 deg**, against the **10.95 deg**
    the pathology actually carries -- mixing the sides cancelled 85% of the asymmetry, and
    the style reward (the largest single weight at 5.0) was therefore asking for a nearly
    symmetric gait.

    Canonicalising both the expert corpus and the agent's own features to one side restores
    it. Left is canonical to match ``reference.py``, which stores its stride left-paretic.
    """
    mirrored = mirror_amp_features(features, mirror_index, mirror_sign)
    return torch.where(is_right_paretic.unsqueeze(-1), mirrored, features)


def extract_amp_features(
    root_pos_z: torch.Tensor,  # (N, 1) or (N,)
    projected_gravity: torch.Tensor,  # (N, 3)
    root_lin_vel: torch.Tensor,  # (N, 3)
    root_ang_vel: torch.Tensor,  # (N, 3)
    joint_pos: torch.Tensor,  # (N, 19)
    joint_vel: torch.Tensor,  # (N, 19)
) -> torch.Tensor:
    """
    Extracts pure kinematic motion features for the AMP discriminator.
    Notice the asymmetry: task conditioning, gait phase, and action history are omitted
    to allow the discriminator to evaluate motion fidelity objectively.
    """
    if root_pos_z.ndim == 1:
        root_pos_z = root_pos_z.unsqueeze(-1)

    features = torch.cat(
        [
            root_pos_z,
            projected_gravity,
            root_lin_vel,
            root_ang_vel,
            joint_pos,
            joint_vel,
        ],
        dim=-1,
    )

    assert features.shape[-1] == AMP_FEATURE_DIM, f"Expected {AMP_FEATURE_DIM}, got {features.shape[-1]}"
    return features


# ==============================================================================
# 2. Discriminator Neural Network Architecture
# ==============================================================================


class AMPDiscriminator(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Discriminator D_psi: (s_t, s_{t+1}) -> R.
    Evaluates kinematic realism of simulated motions relative to clinical stroke priors.
    """

    def __init__(
        self,
        input_dim: int = AMP_TRANSITION_DIM,
        hidden_dims: tuple[int, ...] = (1024, 512),
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim

        # Running normalizer for input feature stability
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.register_buffer("count", torch.tensor(1e-4))

        act_cls = nn.ReLU if activation.lower() == "relu" else nn.ELU

        layers = []
        in_d = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_d, h_dim))
            layers.append(act_cls())
            in_d = h_dim

        # Linear output head (unbounded logit for LSGAN / WGAN)
        layers.append(nn.Linear(in_d, 1))
        self.network = nn.Sequential(*layers)

        # Initialize network weights
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, amp_transitions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            amp_transitions: (batch_size, AMP_TRANSITION_DIM) concatenated (s_t, s_{t+1}).
        Returns:
            scores: (batch_size, 1) real-valued discriminator predictions.
        """
        norm_transitions = (amp_transitions - self.running_mean) / torch.sqrt(self.running_var + 1e-8)
        return self.network(norm_transitions)

    def update_normalizer(self, amp_transitions: torch.Tensor):
        """Updates online mean and variance statistics for feature normalization."""
        with torch.no_grad():
            batch_mean = torch.mean(amp_transitions, dim=0)
            batch_var = torch.var(amp_transitions, dim=0, unbiased=False)
            batch_count = amp_transitions.shape[0]

            delta = batch_mean - self.running_mean
            tot_count = self.count + batch_count

            new_mean = self.running_mean + delta * batch_count / tot_count
            m_a = self.running_var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
            new_var = M2 / tot_count

            self.running_mean.copy_(new_mean)
            self.running_var.copy_(new_var)
            self.count.copy_(tot_count)

    def compute_amp_reward(self, amp_transitions: torch.Tensor) -> torch.Tensor:
        """
        Computes style reward for the RL policy:
          r_amp = max(0, 1 - 0.25 * (D_psi(s, s') - 1)^2)
        """
        with torch.no_grad():
            disc_score = self.forward(amp_transitions).squeeze(-1)
            reward = torch.clamp(1.0 - 0.25 * torch.square(disc_score - 1.0), min=0.0)
            return reward


# ==============================================================================
# 3. AMP Loss and Optimization Manager
# ==============================================================================


class AMPLossManager:
    """
    Manages LSGAN loss calculation with 1-sided gradient penalty on expert samples.
    """

    def __init__(
        self,
        discriminator: AMPDiscriminator,
        learning_rate: float = 1e-4,
        gradient_penalty_weight: float = 5.0,
        weight_decay: float = 1e-4,
    ):
        self.discriminator = discriminator
        self.gp_weight = gradient_penalty_weight
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train_step(
        self,
        expert_transitions: torch.Tensor,
        agent_transitions: torch.Tensor,
    ) -> dict[str, float]:
        # Update normalization stats with combined batch
        combined = torch.cat([expert_transitions, agent_transitions], dim=0)
        self.discriminator.update_normalizer(combined)

        expert_transitions.requires_grad_(True)
        expert_scores = self.discriminator(expert_transitions)
        agent_scores = self.discriminator(agent_transitions.detach())

        # 1. Least-Squares GAN Loss (Demo target = 1.0, Agent target = -1.0)
        loss_demo = 0.5 * torch.mean(torch.square(expert_scores - 1.0))
        loss_agent = 0.5 * torch.mean(torch.square(agent_scores + 1.0))
        lsgan_loss = loss_demo + loss_agent

        # 2. Gradient Penalty on Expert Demonstrations
        ones = torch.ones_like(expert_scores)
        grad_demo = torch.autograd.grad(
            outputs=expert_scores,
            inputs=expert_transitions,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_penalty = 0.5 * self.gp_weight * torch.mean(torch.sum(torch.square(grad_demo), dim=-1))

        total_loss = lsgan_loss + grad_penalty

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=10.0)
        self.optimizer.step()

        with torch.no_grad():
            expert_acc = (expert_scores > 0.0).float().mean().item()
            agent_acc = (agent_scores < 0.0).float().mean().item()

        return {
            "disc_total_loss": total_loss.item(),
            "disc_lsgan_loss": lsgan_loss.item(),
            "disc_gp_loss": grad_penalty.item(),
            "expert_score_mean": expert_scores.mean().item(),
            "agent_score_mean": agent_scores.mean().item(),
            "expert_acc": expert_acc,
            "agent_acc": agent_acc,
        }


# ==============================================================================
# 4. Expert Motion & Agent Replay Buffers
# ==============================================================================


class AMPExpertMotionBuffer:
    """
    Stores and samples contiguous motion transitions (s_t, s_{t+1}) from retargeted
    clinical stroke gait trajectories.
    """

    def __init__(self, dataset_path: str | None = None, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.trajectories: list[torch.Tensor] = []

        if dataset_path and Path(dataset_path).exists():
            self._load_from_npz(dataset_path)
        else:
            # Fall back to whatever clinical data this extension has staged. If nothing is
            # staged either, the synthetic generator below still yields a usable prior.
            from humanoid_pathological_gait.tasks.humanoid_pathological_gait.assets import (
                AMP_EXPERT_CORPUS_FILE,
                EXPERT_DATASET_FILE,
                REFERENCE_STRIDE_FILE,
                resolve_data_file,
            )

            for file_name in (AMP_EXPERT_CORPUS_FILE, REFERENCE_STRIDE_FILE, EXPERT_DATASET_FILE):
                try:
                    self._load_from_npz(str(resolve_data_file(file_name)))
                    if len(self.trajectories) > 0:
                        break
                except (FileNotFoundError, KeyError, ValueError):
                    pass
        if len(self.trajectories) == 0:
            self._generate_synthetic_stroke_trajectories()
        self.constant_feature_dims = self._warn_about_constant_features()

    #: Control period of this task, matching ``sim.dt * decimation``. The discriminator
    #: scores ``(s_t, s_t+1)`` pairs, so expert frames must be this far apart in time or the
    #: expert's state-to-state delta differs from the agent's by the ratio of the two
    #: sampling intervals -- a feature no policy can ever match.
    CONTROL_DT_S = 0.02

    def _load_from_npz(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        if "stride_offsets" in data:
            self._load_expert_corpus(data, npz_path)
        elif "q_trajectory" in data:
            self._load_reference_stride(data, npz_path)
        elif "LHip" in data:
            self._load_raw_clinical(data, npz_path)

    def _load_expert_corpus(self, data, npz_path: str):
        """Loads a retargeted expert corpus -- the preferred prior.

        Written by ``python -m data.batch_parse_gait --solver gmr --corpus`` in the
        ``sw-humanoid-strokegait`` pipeline. Every stride is already retargeted onto this
        robot, sampled at the control rate, and carries the floating-base state the
        retargeter solved, so all forty-eight feature dimensions vary the way the agent's
        do. See ``docs/data.md``.
        """
        offsets = np.asarray(data["stride_offsets"])
        q_all = torch.tensor(np.asarray(data["q"]), dtype=torch.float32, device=self.device)
        v_all = torch.tensor(np.asarray(data["v"]), dtype=torch.float32, device=self.device)
        height = torch.tensor(np.asarray(data["root_height"]), dtype=torch.float32, device=self.device)
        gravity = torch.tensor(np.asarray(data["projected_gravity"]), dtype=torch.float32, device=self.device)
        lin_vel = torch.tensor(np.asarray(data["root_lin_vel"]), dtype=torch.float32, device=self.device)
        ang_vel = torch.tensor(np.asarray(data["root_ang_vel"]), dtype=torch.float32, device=self.device)

        # Canonicalise every stride to a left-paretic frame. Without this the corpus mixes
        # left- and right-impaired patients in raw left/right joint slots and the aggregate
        # asymmetry all but cancels -- see :func:`to_paretic_frame`.
        #
        # The side must come from ``impaired_sides``, which is inferred from the motion, and
        # never from ``paretic_sides``, which derives from the dataset's own labels. The two
        # agree on only 15.5% of strides and only the inferred one puts the pathology the
        # right way round: paretic knee ROM 36.66 deg against a sound 47.61, where the label
        # field gives 46.71 against 37.55 -- backwards. Strides whose side could not be
        # inferred are dropped rather than left unmirrored, which would put symmetric noise
        # back into the prior.
        sides = np.asarray(data["impaired_sides"]).astype(str) if "impaired_sides" in data else None
        mirror_index = mirror_sign = None
        if sides is not None:
            from humanoid_pathological_gait.tasks.humanoid_pathological_gait.h1_joints import H1JointLayout

            layout = H1JointLayout.from_sim_names(
                [str(name) for name in np.asarray(data["joint_names"])], device=self.device
            )
            mirror_index, mirror_sign = layout.mirror_index, layout.mirror_sign

        skipped = 0
        for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
            sl = slice(int(start), int(stop))
            features = extract_amp_features(
                height[sl], gravity[sl], lin_vel[sl], ang_vel[sl], q_all[sl], v_all[sl]
            )
            if sides is not None:
                side = sides[index]
                if side not in ("left", "right"):
                    skipped += 1
                    continue
                if side == "right":
                    features = mirror_amp_features(features, mirror_index, mirror_sign)
            self.trajectories.append(features)

        note = "" if sides is None else f", {skipped} dropped for an un-inferable paretic side"
        frame = "raw left/right slots -- NOT canonicalised" if sides is None else "canonical left-paretic frame"
        print(
            f"[AMPExpertMotionBuffer] Loaded retargeted corpus from {npz_path}: "
            f"{len(self.trajectories)} strides, {int(offsets[-1])} frames "
            f"at {1000 * float(data['control_dt_s']):.0f} ms ({frame}{note})."
        )

    def _load_reference_stride(self, data, npz_path: str):
        """Loads a single retargeted stride.

        Usable, but a corpus of one. When the archive carries a floating base -- the GMR arm
        resolves one, the joint-space baseline does not -- it is used; otherwise the root
        block falls back to constants and :meth:`_warn_about_constant_features` will say so.
        """
        q = torch.tensor(np.asarray(data["q_trajectory"]), dtype=torch.float32, device=self.device)
        v = torch.tensor(np.asarray(data["v_trajectory"]), dtype=torch.float32, device=self.device)
        duration = float(data["stride_duration_s"]) if "stride_duration_s" in data else 1.2
        q, v = self._resample_pair(q, v, duration)
        t_len = q.shape[0]

        if "root_translation" in data and "root_quaternion" in data:
            translation = np.asarray(data["root_translation"], dtype=np.float64)
            quaternion = np.asarray(data["root_quaternion"], dtype=np.float64)
            kinematics = self._root_kinematics(translation, quaternion, duration, t_len)
            feat = extract_amp_features(*kinematics, q, v)
        else:
            feat = extract_amp_features(*self._constant_root_block(t_len), q, v)
        self.trajectories.append(feat)
        print(f"[AMPExpertMotionBuffer] Loaded single reference stride from {npz_path}.")

    def _load_raw_clinical(self, data, npz_path: str):
        """Loads raw clinical joint angles -- the last-resort prior.

        This path cannot supply a floating base, because no retargeting has happened: the
        root block is constant and the discriminator can separate on it alone. It is kept
        only so the buffer still yields something when no retargeted corpus is staged.

        The mapping below is written against
        :data:`~humanoid_pathological_gait.tasks.humanoid_pathological_gait.h1_joints.CLINICAL_JOINT_ORDER`.
        It previously transposed hip roll and hip yaw -- writing ad/abduction into the yaw
        slot and transverse rotation into the roll slot -- and used the same multiplier on
        both limbs. Frontal and transverse multipliers must differ by limb: the clinical
        traces are anatomical (adduction and internal rotation positive on both sides) while
        the H1's hip_roll axes are both +X and its hip_yaw axes both +Z.
        """
        print(
            f"[AMPExpertMotionBuffer] WARNING: falling back to raw clinical angles ({npz_path}). "
            "This prior has no floating base, so ten of the forty-eight AMP features are "
            "constant and the discriminator can separate expert from agent on them alone. "
            "Stage a retargeted corpus (amp_expert_corpus.npz) instead."
        )
        num_strides = min(len(data["subject_ids"]), 50)
        deg2rad = math.pi / 180.0
        durations = np.asarray(data["stride_durations_s"]) if "stride_durations_s" in data else None

        for s_idx in range(num_strides):
            t_len = data["LHip"].shape[1]
            q = torch.zeros((t_len, 19), device=self.device)
            trace = lambda key, col: torch.tensor(  # noqa: E731
                np.asarray(data[key][s_idx, :, col]) * deg2rad, dtype=torch.float32, device=self.device
            )
            #                                  left limb: frontal/transverse multiplier -1
            q[:, 0] = -trace("LHip", 2)   # left_hip_yaw   <- internal/external rotation
            q[:, 1] = -trace("LHip", 1)   # left_hip_roll  <- ad/abduction
            q[:, 2] = -trace("LHip", 0)   # left_hip_pitch <- flexion/extension
            q[:, 3] = trace("LKnee", 0)   # left_knee      <- flexion (opposite sagittal sign)
            q[:, 4] = -trace("LAnkle", 0)  # left_ankle    <- dorsiflexion
            #                                  right limb: frontal/transverse multiplier +1
            q[:, 5] = trace("RHip", 2)    # right_hip_yaw
            q[:, 6] = trace("RHip", 1)    # right_hip_roll
            q[:, 7] = -trace("RHip", 0)   # right_hip_pitch
            q[:, 8] = trace("RKnee", 0)   # right_knee
            q[:, 9] = -trace("RAnkle", 0)  # right_ankle

            q[:, 10] = 0.0  # torso
            q[:, 11:15] = torch.tensor([0.0, 0.1, 0.0, 0.3], device=self.device)  # left arm
            q[:, 15:19] = torch.tensor([0.0, -0.1, 0.0, 0.3], device=self.device)  # right arm

            duration = 1.2
            if durations is not None and s_idx < durations.size and np.isfinite(durations[s_idx]):
                duration = float(durations[s_idx])
            dt = duration / (t_len - 1)
            v = torch.gradient(q, spacing=(dt,), dim=0)[0]
            q, v = self._resample_pair(q, v, duration)
            self.trajectories.append(extract_amp_features(*self._constant_root_block(q.shape[0]), q, v))

    # -- helpers ------------------------------------------------------------

    def _resample_pair(self, q: torch.Tensor, v: torch.Tensor, duration_s: float):
        """Resamples a stride onto this task's control rate."""
        target = max(int(round(duration_s / self.CONTROL_DT_S)) + 1, 8)
        if q.shape[0] == target:
            return q, v
        source_grid = torch.linspace(0.0, 1.0, q.shape[0], device=self.device)
        target_grid = torch.linspace(0.0, 1.0, target, device=self.device)
        index = torch.searchsorted(source_grid, target_grid).clamp(1, q.shape[0] - 1)
        lower, upper = index - 1, index
        weight = ((target_grid - source_grid[lower]) / (source_grid[upper] - source_grid[lower])).unsqueeze(-1)
        return (
            q[lower] * (1.0 - weight) + q[upper] * weight,
            v[lower] * (1.0 - weight) + v[upper] * weight,
        )

    def _constant_root_block(self, t_len: int):
        """The placeholder root block, for priors that carry no floating base."""
        return (
            torch.ones((t_len, 1), device=self.device) * 1.05,
            torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(t_len, 1),
            torch.tensor([0.6, 0.0, 0.0], device=self.device).repeat(t_len, 1),
            torch.zeros((t_len, 3), device=self.device),
        )

    def _root_kinematics(self, translation: np.ndarray, quaternion: np.ndarray, duration_s: float, t_len: int):
        """Body-frame root state from a retargeted floating base, resampled to ``t_len``."""
        from scipy.spatial.transform import Rotation

        def resample(values):
            source = np.linspace(0.0, 1.0, values.shape[0])
            target = np.linspace(0.0, 1.0, t_len)
            return np.stack([np.interp(target, source, values[:, c]) for c in range(values.shape[1])], axis=-1)

        translation = resample(np.asarray(translation, dtype=np.float64))
        quaternion = resample(np.asarray(quaternion, dtype=np.float64))
        quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
        rotations = Rotation.from_quat(np.column_stack([quaternion[:, 1:], quaternion[:, 0]])).as_matrix()

        dt = duration_s / max(t_len - 1, 1)
        linear = np.einsum("tji,tj->ti", rotations, np.gradient(translation, dt, axis=0))
        relative = np.einsum("tji,tjk->tik", rotations[:-1], rotations[1:])
        angular = np.zeros_like(translation)
        angular[:-1] = Rotation.from_matrix(relative).as_rotvec() / dt
        if len(angular) > 1:
            angular[-1] = angular[-2]
        gravity = np.einsum("tji,j->ti", rotations, np.array([0.0, 0.0, -1.0]))

        as_tensor = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)  # noqa: E731
        return (
            as_tensor(translation[:, 2:3]),
            as_tensor(gravity),
            as_tensor(linear),
            as_tensor(angular),
        )

    def _warn_about_constant_features(self) -> list[int]:
        """Reports feature dimensions that never vary across the expert set.

        A constant expert dimension is a free win for the discriminator: the agent's value
        of that dimension is continuous and essentially never lands on the constant, so a
        hyperplane separates the two perfectly and the AMP style reward stops carrying
        gradient. This is the diagnostic that would have caught the hard-coded root block,
        so it runs at construction and says so out loud.
        """
        if not self.trajectories:
            return []
        stacked = torch.cat(self.trajectories, dim=0)
        constant = torch.nonzero(stacked.std(dim=0) < 1e-8).flatten().tolist()
        if constant:
            print(
                f"[AMPExpertMotionBuffer] WARNING: {len(constant)} of {stacked.shape[1]} expert feature "
                f"dimensions are constant (indices {constant}). The discriminator can separate "
                "expert from agent on these alone, which flattens the AMP gradient."
            )
        return constant

    def _generate_synthetic_stroke_trajectories(self, num_trajs: int = 10, traj_len: int = 200):
        for i in range(num_trajs):
            t = torch.linspace(0, 1.2, traj_len, device=self.device)
            phi = 2 * math.pi * t / 1.2
            phi_pi = phi + math.pi

            q = torch.zeros((traj_len, 19), device=self.device)
            q[:, 10] = 0.0  # Torso
            q[:, 11:15] = torch.tensor([0.0, 0.1, 0.0, 0.3], device=self.device)  # Left Arm
            q[:, 15:19] = torch.tensor([0.0, -0.1, 0.0, 0.3], device=self.device)  # Right Arm
            v = torch.zeros((traj_len, 19), device=self.device)

            # Healthy Kinematics
            h_hip = -0.2 + 0.4 * torch.sin(phi)
            h_knee = 0.4 + 0.5 * torch.clamp(torch.sin(phi), min=0.0)
            h_ankle = -0.2 - 0.2 * torch.sin(phi)

            h_hip_pi = -0.2 + 0.4 * torch.sin(phi_pi)
            h_knee_pi = 0.4 + 0.5 * torch.clamp(torch.sin(phi_pi), min=0.0)
            h_ankle_pi = -0.2 - 0.2 * torch.sin(phi_pi)

            # Paretic Kinematics
            p_hip = -0.2 + 0.3 * torch.sin(phi)
            p_knee = 0.2 + 0.15 * torch.sin(phi)
            p_ankle = -0.35 + 0.05 * torch.sin(phi)

            p_hip_pi = -0.2 + 0.3 * torch.sin(phi_pi)
            p_knee_pi = 0.2 + 0.15 * torch.sin(phi_pi)
            p_ankle_pi = -0.35 + 0.05 * torch.sin(phi_pi)

            if i % 2 == 0:
                q[:, 2], q[:, 3], q[:, 4] = p_hip, p_knee, p_ankle
                q[:, 7], q[:, 8], q[:, 9] = h_hip_pi, h_knee_pi, h_ankle_pi
            else:
                q[:, 2], q[:, 3], q[:, 4] = h_hip, h_knee, h_ankle
                q[:, 7], q[:, 8], q[:, 9] = p_hip_pi, p_knee_pi, p_ankle_pi

            v = torch.gradient(q, spacing=(0.02,), dim=0)[0]
            z_root = 1.05 + 0.03 * torch.sin(2 * phi).unsqueeze(-1)
            proj_g = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(traj_len, 1)
            lin_v = torch.column_stack([0.6 + 0.1 * torch.sin(phi), torch.zeros_like(phi), torch.zeros_like(phi)])
            ang_v = torch.zeros((traj_len, 3), device=self.device)

            feat = extract_amp_features(z_root, proj_g, lin_v, ang_v, q, v)
            self.trajectories.append(feat)

    def sample_transitions(self, batch_size: int) -> torch.Tensor:
        transitions = []
        for _ in range(batch_size):
            traj_idx = np.random.randint(len(self.trajectories))
            traj = self.trajectories[traj_idx]
            t_idx = np.random.randint(traj.shape[0] - 1)
            s_t = traj[t_idx]
            s_tp1 = traj[t_idx + 1]
            transitions.append(torch.cat([s_t, s_tp1], dim=-1))

        return torch.stack(transitions, dim=0)


class AMPAgentReplayBuffer:
    """Circular replay buffer for storing simulated transitions collected from policy rollouts."""

    def __init__(self, capacity: int = 50000, device: str | torch.device = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.buffer = torch.zeros((capacity, AMP_TRANSITION_DIM), dtype=torch.float32, device=self.device)
        self.ptr = 0
        self.size = 0

    def insert(self, transitions: torch.Tensor):
        num_items = transitions.shape[0]
        transitions = transitions.to(self.device)

        if self.ptr + num_items <= self.capacity:
            self.buffer[self.ptr : self.ptr + num_items] = transitions
            self.ptr = (self.ptr + num_items) % self.capacity
        else:
            first_part = self.capacity - self.ptr
            second_part = num_items - first_part
            self.buffer[self.ptr :] = transitions[:first_part]
            self.buffer[:second_part] = transitions[first_part:]
            self.ptr = second_part

        self.size = min(self.size + num_items, self.capacity)

    def sample(self, batch_size: int) -> torch.Tensor:
        assert self.size > 0, "Cannot sample from empty replay buffer."
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.buffer[indices]


if __name__ == "__main__":
    print("Testing AMP Discriminator, LSGAN Loss Manager, and Replay Buffers...")
    device = torch.device("cpu")

    disc = AMPDiscriminator(input_dim=AMP_TRANSITION_DIM, hidden_dims=(256, 128))
    loss_mgr = AMPLossManager(disc, learning_rate=1e-4)
    expert_buf = AMPExpertMotionBuffer(device=device)
    agent_buf = AMPAgentReplayBuffer(capacity=1000, device=device)

    synthetic_agent_trans = torch.randn(128, AMP_TRANSITION_DIM, device=device)
    agent_buf.insert(synthetic_agent_trans)

    expert_batch = expert_buf.sample_transitions(batch_size=64)
    agent_batch = agent_buf.sample(batch_size=64)

    metrics = loss_mgr.train_step(expert_batch, agent_batch)
    print("AMP Discriminator Training Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    rewards = disc.compute_amp_reward(expert_batch[:8])
    print(f"AMP Style Rewards (Expert Batch, shape {rewards.shape}):\n  {rewards.tolist()}")
    print("AMP Discriminator module verification passed successfully!")
