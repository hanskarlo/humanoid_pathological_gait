# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The PPO mirror augmentation's signed permutation (algorithms/ppo/symmetry.py), without Isaac Sim."""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

EXT = pathlib.Path(__file__).resolve().parents[1] / "source" / "humanoid_pathological_gait"
sys.path.insert(0, str(EXT))
STRIDE = (
    EXT
    / "humanoid_pathological_gait/tasks/humanoid_pathological_gait/data/references/healthy_s049_stride0_grounded.npz"
)

#: The policy observation as ObservationsCfg.PolicyCfg lays it out (109 dims), plus future frames.
TERMS = [
    ("projected_gravity", 3),
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("joint_pos", 19),
    ("joint_vel", 19),
    ("reference_joint_pos", 19),
    ("reference_joint_vel", 19),
    ("paretic_side", 1),
    ("gait_phase", 2),
    ("root_progression_error", 2),
    ("actions", 19),
]


def _layout():
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.h1_joints import (
        CLINICAL_JOINT_ORDER,
        H1JointLayout,
    )

    # A non-trivial simulation order, so an index/offset slip cannot hide behind the identity.
    names = list(CLINICAL_JOINT_ORDER)[::-1]
    return H1JointLayout.from_sim_names(names, device="cpu")


def _mirror(terms):
    from humanoid_pathological_gait.algorithms.ppo.symmetry import MirrorAugmentation, signed_permutation

    layout = _layout()
    index, sign = signed_permutation(terms, layout.mirror_index, layout.mirror_sign)
    return MirrorAugmentation(index, sign, layout.mirror_index, layout.mirror_sign, "cpu"), layout


def test_the_mirror_is_an_involution_and_sizes_add_up():
    m, _ = _mirror(TERMS + [("reference_future", 42)])
    assert m.obs_index.numel() == 109 + 42
    obs = torch.randn(64, 151)
    assert torch.allclose(m.obs(m.obs(obs)), obs)
    actions = torch.randn(64, 19)
    assert torch.allclose(m.actions(m.actions(actions)), actions)


def test_each_term_mirrors_as_its_physics_says():
    m, layout = _mirror(TERMS)
    obs = torch.randn(4, 109)
    out = m.obs(obs)
    assert torch.equal(out[:, 0:3], obs[:, 0:3] * torch.tensor([1.0, -1.0, 1.0]))  # gravity, polar
    assert torch.equal(out[:, 3:6], obs[:, 3:6] * torch.tensor([1.0, -1.0, 1.0]))  # linear velocity, polar
    assert torch.equal(out[:, 6:9], obs[:, 6:9] * torch.tensor([-1.0, 1.0, -1.0]))  # angular velocity, axial
    assert torch.allclose(out[:, 9:28], layout.mirror(obs[:, 9:28]))  # joint positions
    assert torch.equal(out[:, 85], -obs[:, 85])  # paretic side
    assert torch.equal(out[:, 86:88], obs[:, 86:88])  # phase
    assert torch.equal(out[:, 88:90], obs[:, 88:90] * torch.tensor([1.0, -1.0]))  # along, cross
    assert torch.allclose(out[:, 90:109], layout.mirror(obs[:, 90:109]))  # last action


def test_an_unknown_term_raises():
    from humanoid_pathological_gait.algorithms.ppo.symmetry import signed_permutation

    layout = _layout()
    with pytest.raises(KeyError):
        signed_permutation(TERMS + [("mystery", 3)], layout.mirror_index, layout.mirror_sign)


@pytest.mark.skipif(not STRIDE.exists(), reason="grounded healthy stride not staged")
def test_it_is_the_environments_own_symmetry():
    """Mirroring a left-paretic env's reference terms gives the right-paretic env's, at the same phase."""
    from humanoid_pathological_gait.algorithms.ppo.symmetry import signed_permutation
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.h1_joints import (
        CLINICAL_JOINT_ORDER,
        H1JointLayout,
    )
    from humanoid_pathological_gait.tasks.humanoid_pathological_gait.reference import ReferenceGaitManager

    layout = H1JointLayout.from_sim_names(list(CLINICAL_JOINT_ORDER), device="cpu")
    ref = ReferenceGaitManager(str(STRIDE), layout, num_envs=2, device="cpu")
    ref.gait_phase[:] = 0.37
    ref.paretic_side[:] = torch.tensor([-1.0, 1.0])
    q, v = ref.sample()
    q_future, c_future = ref.sample_ahead(0.1)
    terms = [("reference_joint_pos", 19), ("reference_joint_vel", 19), ("paretic_side", 1), ("reference_future", 21)]
    obs = torch.cat([q, v, ref.paretic_side.unsqueeze(-1), q_future, c_future], dim=-1)
    index, sign = signed_permutation(terms, layout.mirror_index, layout.mirror_sign)
    mirrored_left = obs[0, index] * sign
    assert torch.allclose(mirrored_left, obs[1], atol=1e-6)
