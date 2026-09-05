# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolution of the clinical gait data files this task depends on.

Two data products are needed, both produced by an upstream clinical-motion-capture
pipeline (originally ``sw-humanoid-strokegait``):

* ``retargeted_h1_stride.npz`` -- one Pinocchio-QP-retargeted 19-DoF stride, the
  reference trajectory the policy tracks and resets onto. Required.
* ``amp_expert_corpus.npz`` -- many retargeted strides at this task's control rate, each
  carrying the floating-base state the retargeter solved. This is the AMP expert motion
  prior. Optional but strongly preferred: the fallbacks below cannot supply a floating
  base, which leaves ten of the forty-eight AMP feature dimensions constant and lets the
  discriminator separate expert from agent without looking at the motion at all.
* ``stroke_gait_dataset.npz`` -- the raw parsed post-stroke corpus. Last-resort AMP prior;
  it has no floating base and no retargeting.

Both are looked up under this package's ``data/`` directory first, so a checkout that
stages them runs standalone. ``SW_STROKEGAIT_DATA_DIR`` overrides the search for
site-specific layouts, and a ``data/`` directory beside the extension is the last
fallback -- useful when this extension is embedded as a submodule of the repository
that generates the data.

See ``docs/data.md`` for the file schemas and how to regenerate them.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
"""Directory holding the staged clinical data files."""

REFERENCE_STRIDE_FILE = "retargeted_h1_stride.npz"
"""Retargeted 19-DoF H1 reference stride (1000 samples over 1.2 s)."""

EXPERT_DATASET_FILE = "stroke_gait_dataset.npz"
"""Raw parsed post-stroke gait corpus; the last-resort AMP prior."""

AMP_EXPERT_CORPUS_FILE = "amp_expert_corpus.npz"
"""Retargeted, control-rate AMP expert corpus with a solved floating base."""


def _search_roots() -> list[Path]:
    """Directories searched for a data file, highest priority first."""
    roots = [DATA_DIR]
    override = os.environ.get("SW_STROKEGAIT_DATA_DIR")
    if override:
        roots.append(Path(override))
    # Fallback: a data/ directory beside the extension (e.g. the parent repo of a submodule).
    roots.append(Path(__file__).resolve().parents[6] / "data")
    return roots


def resolve_data_file(file_name: str) -> Path:
    """Return the path of *file_name*, searching the staged and fallback locations.

    Raises:
        FileNotFoundError: If the file is in none of the search roots.
    """
    searched = []
    for root in _search_roots():
        candidate = root / file_name
        searched.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate clinical data file '{file_name}'. Searched:\n  " + "\n  ".join(searched) + "\n"
        "Regenerate it with the outer repo's data pipeline, or point SW_STROKEGAIT_DATA_DIR at a directory holding it."
    )


def reference_stride_path() -> Path:
    """Path to the retargeted H1 reference stride."""
    return resolve_data_file(REFERENCE_STRIDE_FILE)


def expert_dataset_path() -> Path:
    """Path to the raw clinical corpus (the last-resort AMP prior)."""
    return resolve_data_file(EXPERT_DATASET_FILE)


def amp_expert_prior_path() -> Path:
    """Path to the best AMP expert prior available, most preferred first.

    ``amp_expert_corpus.npz`` is the one to use: retargeted onto this robot, sampled at the
    control rate, with a solved floating base. The reference stride is a corpus of one but
    still carries a floating base when the GMR arm produced it. The raw clinical corpus is
    the last resort and leaves ten AMP feature dimensions constant.
    """
    for file_name in (AMP_EXPERT_CORPUS_FILE, REFERENCE_STRIDE_FILE, EXPERT_DATASET_FILE):
        try:
            return resolve_data_file(file_name)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "No AMP expert prior found. Generate one with "
        "`python -m data.batch_parse_gait --solver gmr --corpus` in sw-humanoid-strokegait."
    )
