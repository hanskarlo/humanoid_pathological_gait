#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preflight check on the staged clinical data, before a long run consumes it.

Everything here is a property that has silently been wrong at some point in this repo's
history and cost a training run to discover:

* the reference stride declares which of its two legs carries the pathology, so the
  environment weakens the limb the reference actually walks stiffly on rather than falling
  back on an assumption in the loader;
* the AMP expert prior resolves to the retargeted corpus and not to a fallback, and no
  feature dimension in it is constant -- a constant dimension lets the discriminator
  separate expert from agent without looking at the motion, which flattens the AMP
  gradient for the whole run;
* the expert prior is sampled at this task's control period, since the discriminator
  scores state *transitions* and a mismatched interval is a feature no policy can match;
* the corpus carries a per-stride impaired side inferred from the motion, because the
  dataset's own side labels are inverted for most of the cohort and anything that groups
  by paretic versus sound limb would otherwise cancel the asymmetry it is measuring.

Needs neither Isaac Sim nor a GPU, so it runs in about a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "humanoid_pathological_gait"))

from humanoid_pathological_gait.algorithms.amp import AMPExpertMotionBuffer  # noqa: E402
from humanoid_pathological_gait.tasks.humanoid_pathological_gait.assets import (  # noqa: E402
    AMP_EXPERT_CORPUS_FILE,
    amp_expert_prior_path,
    reference_stride_path,
)
from humanoid_pathological_gait.tasks.humanoid_pathological_gait.h1_joints import (  # noqa: E402
    CLINICAL_JOINT_ORDER,
    H1JointLayout,
)
from humanoid_pathological_gait.tasks.humanoid_pathological_gait.reference import (  # noqa: E402
    ReferenceGaitManager,
)

#: ``sim.dt`` 0.005 x ``decimation`` 4. Kept here rather than imported so that this check
#: fails loudly if the environment's control rate is changed without the corpus being
#: regenerated, instead of silently agreeing with whatever the environment now says.
CONTROL_DT_S = 0.02

failures: list[str] = []


def check(condition: bool, ok_message: str, fail_message: str) -> None:
    if condition:
        print(f"  ok   {ok_message}")
    else:
        print(f"  FAIL {fail_message}")
        failures.append(fail_message)


def main() -> int:
    stride_path = reference_stride_path()
    print(f"reference stride: {stride_path}")
    layout = H1JointLayout.from_sim_names(list(CLINICAL_JOINT_ORDER), device="cpu")
    manager = ReferenceGaitManager(str(stride_path), layout, num_envs=1, device="cpu")
    archive = np.load(stride_path, allow_pickle=True)

    check(
        "impaired_side" in archive.files and str(archive["impaired_side"]) in ("left", "right"),
        f"declares its impaired side ({manager.impaired_side})",
        "does not declare impaired_side; the loader is falling back to assuming 'left'. "
        "Regenerate it with data/batch_parse_gait.py in sw-humanoid-strokegait.",
    )
    check(
        manager.stride_duration_s > 0.0,
        f"carries a measured stride duration ({manager.stride_duration_s:.3f} s)",
        "has no usable stride duration, so reference velocities fall back to a 1.2 s assumption",
    )

    prior_path = amp_expert_prior_path()
    print(f"\nAMP expert prior: {prior_path}")
    check(
        prior_path.name == AMP_EXPERT_CORPUS_FILE,
        "resolves to the retargeted corpus",
        f"resolves to {prior_path.name}, a fallback. The fallbacks have no floating base; "
        "the discriminator will saturate. Stage amp_expert_corpus.npz.",
    )

    buffer = AMPExpertMotionBuffer(dataset_path=str(prior_path), device="cpu")
    features = torch.cat(buffer.trajectories, dim=0)
    check(
        len(buffer.constant_feature_dims) == 0,
        f"has no constant feature dimension across {features.shape[0]} frames",
        f"has {len(buffer.constant_feature_dims)} constant feature dimensions "
        f"{list(buffer.constant_feature_dims)}; the discriminator can separate on those alone",
    )

    corpus = np.load(prior_path, allow_pickle=True)
    check(
        "impaired_sides" in corpus.files,
        "carries a measured impaired side per stride",
        "has no per-stride impaired side. Anything grouping the corpus by paretic versus "
        "sound limb would have to fall back on the dataset's labels, which are inverted "
        "for most of the cohort (see data/paretic_side.py upstream).",
    )
    if "impaired_sides" in corpus.files:
        sides = np.asarray(corpus["impaired_sides"])
        callable_fraction = float((sides != "unknown").mean())
        check(
            callable_fraction > 0.5,
            f"{callable_fraction:.0%} of strides have a side the motion can call "
            f"({int((sides == 'unknown').sum())} too symmetric)",
            f"only {callable_fraction:.0%} of strides could be assigned a side",
        )

    if "stride_durations_s" in corpus and "stride_lengths" in corpus:
        durations = np.asarray(corpus["stride_durations_s"], dtype=np.float64)
        lengths = np.asarray(corpus["stride_lengths"], dtype=np.float64)
        dt = durations / np.maximum(lengths - 1, 1)
        check(
            bool(np.allclose(dt, CONTROL_DT_S, atol=1e-3)),
            f"is sampled at the {CONTROL_DT_S * 1000:.0f} ms control period",
            f"is sampled at {np.median(dt) * 1000:.2f} ms against the task's "
            f"{CONTROL_DT_S * 1000:.0f} ms; the discriminator separates on the interval alone",
        )

    print()
    if failures:
        print(f"{len(failures)} clinical-data check(s) failed.")
        return 1
    print("All clinical-data checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
