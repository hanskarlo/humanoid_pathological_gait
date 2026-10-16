# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve the NVIDIA assets this task loads from a local, checksummed mirror instead of the network.

The H1 USD and the ground plane are fetched from NVIDIA's asset server at every scene build. On
2026-10-15 a network outage at ~04:45 broke an overnight campaign:

* the robot spawner raised (``USD file not found at path: 'https://...'``);
* the ground plane, which is created first, did **not** raise. Offline it would have become a
  plane with an unresolved reference, and a run with only the robot mirrored would have trained
  on no floor without crashing.

So the mirror covers both, and every file the scene reads from the server (see
:data:`MANIFEST_PATH`). :func:`install` points ``omni.client`` at it with
:func:`omni.client.set_alias`, which redirects every read under the server prefix: the stage, the
spawner's existence check (``omni.client.stat``) and the ground plane alike. No config path
changes, so the scene is built from the same bytes. The freeze gate checks that the result
reproduces exactly.

Build the mirror with ``python scripts/mirror_assets.py fetch``. Set ``SW_ASSET_MIRROR=0`` to
disable it. If any file is missing or fails its checksum, :func:`install` leaves the network path
in place and says so.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

#: The server prefix every mirrored URL starts with (Isaac Sim 6.0's asset root).
ASSET_SERVER = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/"
#: Default mirror location: ``<extension root>/assets_mirror`` (gitignored; NVIDIA's files are not
#: redistributed through the repository, only their checksums).
DEFAULT_ROOT = Path(__file__).resolve().parents[5] / "assets_mirror"
#: Relative path under the server prefix -> sha256, for every file the scene reads from the server.
MANIFEST_PATH = Path(__file__).with_name("asset_mirror_manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path) -> list[str]:
    """Problems with the mirror at ``root``; empty when every manifest file is present and intact."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    problems = []
    for rel, expected in manifest.items():
        path = root / rel
        if not path.is_file():
            problems.append(f"missing {rel}")
        elif sha256(path) != expected:
            problems.append(f"checksum mismatch {rel}")
    return problems


def install() -> dict[str, object] | None:
    """Redirect the asset server to the local mirror. Call after Kit starts and before the scene builds.

    Returns what was installed, for the run's provenance, or None if the mirror is disabled,
    absent or damaged (the network path is then used, as before).
    """
    if os.environ.get("SW_ASSET_MIRROR", "1") == "0":
        print("[asset_mirror] disabled (SW_ASSET_MIRROR=0); assets come from the network")
        return None
    root = Path(os.environ.get("SW_ASSET_MIRROR_ROOT", DEFAULT_ROOT))
    problems = verify(root)
    if problems:
        print(f"[asset_mirror] not used ({'; '.join(problems[:3])}); assets come from the network")
        return None
    import omni.client  # noqa: PLC0415 - only importable once Kit is up

    omni.client.set_alias(ASSET_SERVER, root.as_uri() + "/")
    print(f"[asset_mirror] {ASSET_SERVER} -> {root} ({len(json.loads(MANIFEST_PATH.read_text()))} files verified)")
    return {"server": ASSET_SERVER, "root": str(root), "manifest_sha256": sha256(MANIFEST_PATH)}
