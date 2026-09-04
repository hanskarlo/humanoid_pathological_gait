# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolution of the clinical gait data files this task depends on.

Two data products are needed, both produced by an upstream clinical-motion-capture
pipeline (originally ``sw-humanoid-strokegait``):

* ``retargeted_h1_stride.npz`` -- one Pinocchio-QP-retargeted 19-DoF stride, the
  reference trajectory the policy tracks and resets onto. Required.
* ``stroke_gait_dataset.npz`` -- a corpus of parsed post-stroke strides, used as the
  AMP expert motion prior. Optional: without it the AMP buffer falls back to the
  reference stride, and failing that to a synthetic prior.

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
"""Parsed post-stroke gait corpus used as the AMP expert motion prior."""


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
    """Path to the AMP expert motion corpus."""
    return resolve_data_file(EXPERT_DATASET_FILE)
