# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Find which AMP feature dimensions let the discriminator separate expert from agent.

A saturated discriminator (``expert_acc`` pinned at 1.000) has a flat gradient and teaches
the policy nothing, so when a training run saturates the question is *what* it is separating
on. This rolls out a trained checkpoint, collects the agent's AMP features, and scores every
one of the 48 dimensions against the expert prior's.

Two scores per dimension, because they fail differently:

* **Overlap.** The fraction of the agent's mass lying inside the expert's 1st-99th percentile
  range. A dimension with zero overlap is separable by a threshold -- the failure mode the
  constant-feature diagnostic in ``AMPExpertMotionBuffer`` catches at construction, but for
  dimensions that vary in both distributions and still do not intersect.
* **Single-dimension AUC**, from a one-feature logistic fit. Near 0.5 is indistinguishable;
  near 1.0 means this dimension alone tells the two apart.

.. note::
   High accuracy early in training is not by itself pathological -- an undertrained policy
   really does move differently from a stroke patient, and the discriminator is supposed to
   say so. What matters is whether the separation rests on motion (which the policy can fix
   by walking better) or on an artefact of how the expert set was built (which it cannot).
   This script does not distinguish those; it localises the signal so a human can.

Run against a finished run's checkpoint::

    scripts/diagnose_amp_separability.py --checkpoint logs/ppo_amp/<run>/model_1200.pt
"""

import argparse

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="Isaac-H1-Pathological-Gait-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Policy checkpoint to roll out.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=200, help="Rollout steps to collect agent features over.")
parser.add_argument("--amp_prior", type=str, default=None, help="Expert prior; defaults to the best staged one.")
parser.add_argument("--top", type=int, default=12, help="How many worst dimensions to print.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs only once the simulation app is up."""

import importlib  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

importlib.import_module("humanoid_pathological_gait.tasks")

from humanoid_pathological_gait.algorithms.amp import (  # noqa: E402
    AMPExpertMotionBuffer,
    extract_amp_features,
)
from humanoid_pathological_gait.algorithms.ppo import ActorCritic  # noqa: E402
from humanoid_pathological_gait.tasks.humanoid_pathological_gait.assets import (  # noqa: E402
    amp_expert_prior_path,
)

#: Names for the 48 AMP feature dimensions, in the order ``extract_amp_features`` packs them.
FEATURE_NAMES = (
    ["root_height"]
    + [f"proj_gravity_{a}" for a in "xyz"]
    + [f"root_lin_vel_{a}" for a in "xyz"]
    + [f"root_ang_vel_{a}" for a in "xyz"]
    + [f"q[{i}]" for i in range(19)]
    + [f"v[{i}]" for i in range(19)]
)


def overlap_fraction(agent: np.ndarray, expert: np.ndarray) -> float:
    """Fraction of agent samples inside the expert's 1st-99th percentile band."""
    lo, hi = np.percentile(expert, [1.0, 99.0])
    if hi <= lo:  # a constant expert dimension: any spread at all separates
        return float(np.mean(np.isclose(agent, lo)))
    return float(np.mean((agent >= lo) & (agent <= hi)))


def single_feature_auc(agent: np.ndarray, expert: np.ndarray) -> float:
    """AUC of the best threshold on one dimension, computed by rank statistics.

    Reported folded to [0.5, 1.0]: it does not matter which side the expert falls on, only
    whether a threshold exists at all.
    """
    values = np.concatenate([expert, agent])
    labels = np.concatenate([np.ones(expert.size), np.zeros(agent.size)])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1)
    positive = labels == 1
    auc = (ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (
        positive.sum() * (~positive).sum()
    )
    return float(max(auc, 1.0 - auc))


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    device = env.device

    obs, _ = env.reset()
    obs = obs["policy"]
    policy = ActorCritic(obs.shape[-1], env.action_space.shape[-1]).to(device)
    checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["policy"] if "policy" in checkpoint else checkpoint)
    policy.eval()

    collected = []
    with torch.no_grad():
        for _ in range(args_cli.steps):
            action = policy.act_inference(obs) if hasattr(policy, "act_inference") else policy(obs)[0]
            obs, _, _, _, _ = env.step(action)
            obs = obs["policy"]
            collected.append(extract_amp_features(*env.get_amp_kinematic_tensors()).cpu().numpy())
    agent = np.concatenate(collected, axis=0)

    prior = args_cli.amp_prior or str(amp_expert_prior_path())
    buffer = AMPExpertMotionBuffer(dataset_path=prior, device="cpu")
    expert = torch.cat(buffer.trajectories, dim=0).cpu().numpy()

    print(f"\nagent {agent.shape[0]} samples from {args_cli.checkpoint}")
    print(f"expert {expert.shape[0]} samples from {prior}\n")

    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        rows.append((overlap_fraction(agent[:, i], expert[:, i]), single_feature_auc(agent[:, i], expert[:, i]), i, name))
    rows.sort(key=lambda r: (r[0], -r[1]))

    print("Most separable dimensions (lowest agent-inside-expert-range overlap first):")
    print(f"  {'dim':>3s} {'name':16s} {'overlap':>8s} {'auc':>6s} {'agent mean':>11s} {'expert mean':>12s}")
    for overlap, auc, i, name in rows[: args_cli.top]:
        print(
            f"  {i:3d} {name:16s} {overlap:8.3f} {auc:6.3f} {agent[:, i].mean():11.4f} {expert[:, i].mean():12.4f}"
        )

    disjoint = [r for r in rows if r[0] < 0.01]
    print(
        f"\n{len(disjoint)} of 48 dimensions have <1% overlap with the expert range"
        + (f": {[r[2] for r in disjoint]}" if disjoint else " -- none, so no dimension separates alone.")
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
