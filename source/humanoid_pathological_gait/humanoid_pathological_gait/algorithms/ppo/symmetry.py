"""Left/right mirror symmetry for PPO data augmentation (forward plan 2026-10-13, Phase 2b).

Mittal et al., "Symmetry considerations for learning task symmetric robot policies", ICRA 2024
(arXiv 2403.04359), as rsl_rl implements it: every PPO mini-batch is extended with its mirror image
-- mirrored observations and actions, with the advantages, returns and old log-probabilities
repeated. The task is mirror-equivariant by construction: a right-paretic environment is the mirror
of a left-paretic one in its reference, impairment, randomisation, pushes and (since the
2026-09-24 sway fix) its root path. Yet one policy's two sides settle on different gaits, up to 21
points apart in weight bearing, which inflates seed variance.

The mirror is a **signed permutation** of the observation vector, built from the observation
manager's own term list, so a term this module does not know raises instead of being guessed. The
frame is the robot's base frame, x forward, y left, z up, mirrored across the x-z plane:

* polar vectors (projected gravity, linear velocity): ``(x, -y, z)``;
* the angular velocity, an axial vector: ``(-x, y, -z)``;
* joint-space terms: the joint layout's own ``mirror_index`` / ``mirror_sign`` (the same map the
  reference's mirrored tables are built with);
* ``paretic_side`` negates. The phase, and contact schedules already in paretic/sound order, are
  unchanged: flipping the side keeps "paretic" meaning the paretic limb;
* the root-progression error: along-track unchanged, cross-track negated.

Pure torch: testable without Isaac Sim (``tests/test_symmetry.py``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

_POLAR = (1.0, -1.0, 1.0)
_AXIAL = (-1.0, 1.0, -1.0)
#: Observation terms that are joint-space vectors (19, simulation order).
JOINT_TERMS = ("joint_pos", "joint_vel", "reference_joint_pos", "reference_joint_vel", "actions")


def signed_permutation(
    terms: Sequence[tuple[str, int]], mirror_index: torch.Tensor, mirror_sign: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(index, sign)`` with ``mirrored = obs[..., index] * sign``, for an observation laid out as ``terms``.

    Args:
        terms: ``(name, dim)`` per observation term, in order.
        mirror_index, mirror_sign: the joint layout's mirror (19 joints, simulation order).

    Raises:
        KeyError: for a term whose mirror is not defined here.
        ValueError: when a term's dimension does not match its known layout.
    """
    n_joints = int(mirror_index.numel())
    index, sign = [], []
    offset = 0
    for name, dim in terms:
        local: list[int]
        signs: list[float]
        if name in ("projected_gravity", "base_lin_vel", "base_ang_vel"):
            if dim != 3:
                raise ValueError(f"{name} has dim {dim}, expected 3")
            local, signs = [0, 1, 2], list(_AXIAL if name == "base_ang_vel" else _POLAR)
        elif name in JOINT_TERMS:
            if dim != n_joints:
                raise ValueError(f"{name} has dim {dim}, expected {n_joints}")
            local, signs = mirror_index.tolist(), mirror_sign.tolist()
        elif name == "paretic_side":
            local, signs = [0], [-1.0]
        elif name == "gait_phase":
            local, signs = list(range(dim)), [1.0] * dim
        elif name == "root_progression_error":
            if dim != 2:
                raise ValueError(f"{name} has dim {dim}, expected 2")
            local, signs = [0, 1], [1.0, -1.0]
        elif name == "reference_future":
            # Per lead: the reference pose (joint space) then its contact schedule (paretic, sound).
            block = n_joints + 2
            if dim % block:
                raise ValueError(f"{name} has dim {dim}, not a multiple of {block}")
            local, signs = [], []
            for lead in range(dim // block):
                base = lead * block
                local += [base + i for i in mirror_index.tolist()] + [base + n_joints, base + n_joints + 1]
                signs += mirror_sign.tolist() + [1.0, 1.0]
        else:
            raise KeyError(f"no mirror defined for observation term {name!r}; add it before using symmetry")
        index += [offset + i for i in local]
        sign += signs
        offset += dim
    return torch.tensor(index, dtype=torch.long), torch.tensor(sign, dtype=torch.float32)


class MirrorAugmentation:
    """Mirror an observation batch and an action batch. Built once per environment."""

    def __init__(self, obs_index, obs_sign, action_index, action_sign, device):
        self.obs_index = obs_index.to(device)
        self.obs_sign = obs_sign.to(device)
        self.action_index = action_index.to(device)
        self.action_sign = action_sign.to(device)

    @classmethod
    def from_env(cls, env, group: str = "policy") -> MirrorAugmentation:
        manager = env.observation_manager
        names = list(manager.active_terms[group])
        dims = [int(torch.tensor(d).prod()) for d in manager.group_obs_term_dim[group]]
        layout = env.joint_layout
        obs_index, obs_sign = signed_permutation(
            list(zip(names, dims)), layout.mirror_index.cpu(), layout.mirror_sign.cpu()
        )
        return cls(obs_index, obs_sign, layout.mirror_index.cpu(), layout.mirror_sign.cpu(), env.device)

    def obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[..., self.obs_index] * self.obs_sign

    def actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions[..., self.action_index] * self.action_sign
