#!/usr/bin/env python3
"""Fetch the NVIDIA assets the task reads from the network into a local, checksummed mirror.

See ``tasks/humanoid_pathological_gait/asset_mirror.py`` for why. The file list is every file the
scene reads from the asset server: the H1 USD and its meshes, the ground plane and its textures.
It was found by walking each USD's dependencies with ``pxr`` (2026-10-15).

    python scripts/mirror_assets.py fetch     # download into assets_mirror/ and verify against the manifest
    python scripts/mirror_assets.py verify    # check an existing mirror
    python scripts/mirror_assets.py fetch --write-manifest   # (re)record the checksums; only on purpose

Plain Python (no Isaac Sim), so it runs before the simulator does.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

EXT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "asset_mirror",
    EXT
    / "source/humanoid_pathological_gait/humanoid_pathological_gait/tasks/humanoid_pathological_gait/asset_mirror.py",
)
asset_mirror = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(asset_mirror)

FILES = (
    "Isaac/IsaacLab/Robots/Unitree/H1/h1_minimal.usd",
    "Isaac/IsaacLab/Robots/Unitree/H1/instanceable_meshes.usd",
    "Isaac/Environments/Grid/default_environment.usd",
    "Isaac/Environments/Grid/Materials/Textures/WireframeBlur_basecolor.png",
    "Isaac/Environments/Grid/Materials/Textures/WireframeBlur_blue.png",
    "Isaac/Environments/Grid/Materials/Textures/Wireframe_blue.png",
)


def fetch(root: Path) -> None:
    for rel in FILES:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(asset_mirror.ASSET_SERVER + rel, timeout=120) as response:
            target.write_bytes(response.read())
        print(f"  {rel}: {target.stat().st_size} bytes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("fetch", "verify"))
    parser.add_argument("--root", type=Path, default=asset_mirror.DEFAULT_ROOT)
    parser.add_argument("--write-manifest", action="store_true", help="record the fetched files' checksums")
    args = parser.parse_args()
    if args.mode == "fetch":
        print(f"fetching {len(FILES)} files from {asset_mirror.ASSET_SERVER} into {args.root}")
        fetch(args.root)
        if args.write_manifest:
            manifest = {rel: asset_mirror.sha256(args.root / rel) for rel in FILES}
            asset_mirror.MANIFEST_PATH.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
            print(f"wrote {asset_mirror.MANIFEST_PATH.name}")
    problems = asset_mirror.verify(args.root)
    for line in problems:
        print(f"  PROBLEM {line}")
    print("mirror OK" if not problems else f"mirror NOT usable: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
