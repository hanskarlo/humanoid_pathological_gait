# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the supplementary video and the gait filmstrip for a trained checkpoint.

``play.py --video`` records a fixed Kit camera, which the robot walks straight out of
within a few seconds. This follows one robot with :class:`~gait_camera.TrackingCamera`,
records the canonical gait views, and additionally exports a **filmstrip**: evenly
spaced frames across exactly one gait cycle, tiled into a single image. That figure --
a stride shown as a row of postures -- is what a reader looks at to see foot drop and
reduced knee flexion, and it cannot be read off a plot.

Recording starts after a warmup so the reset transient is not the first thing on screen,
and the clip is trimmed to a whole number of gait cycles so it loops cleanly.

Examples::

    # Supplementary video, sagittal view of one left-paretic subject
    uv run python scripts/record_video.py --checkpoint logs/ppo_amp/<run>/model_3500.pt

    # All four canonical views plus the one-cycle filmstrip
    uv run python scripts/record_video.py --checkpoint <ckpt> --views sagittal frontal oblique --filmstrip

.. note::
    The simulation app is launched before any task import; see ``zero_agent.py`` for why.
    ``--enable_cameras`` is forced on: rendering is the entire point of this script.
"""

import argparse

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a train_amp.py checkpoint (.pt).")
parser.add_argument("--task", type=str, default="Isaac-H1-Pathological-Gait-Play-v0")
parser.add_argument("--num_envs", type=int, default=1, help="Environments to simulate; only one is filmed.")
parser.add_argument("--env_index", type=int, default=0, help="Which environment the camera follows.")
parser.add_argument(
    "--paretic_side",
    type=str,
    default="left",
    choices=("left", "right"),
    help="Paretic side of the filmed robot. The reference stride is left-paretic; 'right' films its mirror.",
)
parser.add_argument(
    "--views",
    type=str,
    nargs="+",
    default=["sagittal"],
    help="Camera views to record, one clip each: sagittal, frontal, oblique, wide.",
)
parser.add_argument("--num_cycles", type=float, default=6.0, help="Gait cycles to record per view.")
parser.add_argument("--warmup_steps", type=int, default=150, help="Steps to run before recording starts.")
parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 1080], help="Frame size, width height.")
parser.add_argument("--filmstrip", action="store_true", help="Also export a one-cycle filmstrip image.")
parser.add_argument("--filmstrip_frames", type=int, default=8, help="Postures tiled across the filmstrip.")
parser.add_argument("--output_dir", type=str, default=None, help="Defaults to <checkpoint dir>/videos.")
parser.add_argument(
    "--presets", type=str, nargs="*", default=(), help="Preset variants to select, e.g. --presets newton_mjwarp."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Rendering is the point of this script, so the cameras are not optional.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs only once the simulation app is up."""

import importlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import load_cfg_from_registry, resolve_presets

importlib.import_module("humanoid_pathological_gait.tasks")

from gait_camera import VIEWS, TrackingCamera, configure_viewer  # noqa: E402

from humanoid_pathological_gait.algorithms.ppo import ActorCritic  # noqa: E402


def drop_warmup_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Discard the black frames the Kit renderer emits before its render product is live.

    ``IsaacsimKitPerspectiveVideo`` returns an all-zero frame while the replicator
    annotator is still warming up, which puts a black flash at the head of the clip and,
    worse, a black panel at the left of the filmstrip.
    """
    first = next((index for index, frame in enumerate(frames) if frame.mean() > 1.0), len(frames))
    if first:
        print(f"[record_video] dropped {first} blank warmup frame(s)")
    return frames[first:]


def write_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    """Encode frames to MP4, falling back to a GIF when no MP4 writer is available."""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Even dimensions and yuv420p: H.264 requires both, and a video that PowerPoint
        # and the IEEE submission portal will not open is not a deliverable.
        cropped = [frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2] for frame in frames]
        imageio.mimwrite(path, cropped, fps=fps, quality=8, macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
        print(f"[record_video] wrote {path} ({len(frames)} frames @ {fps:.1f} fps)")
    except Exception as error:  # noqa: BLE001 - any encoder failure should still yield a file
        fallback = path.with_suffix(".gif")
        imageio.mimwrite(fallback, frames, fps=fps)
        print(f"[record_video] MP4 encoding failed ({error}); wrote {fallback} instead")


def write_filmstrip(frames: list[np.ndarray], path: Path, columns: int) -> None:
    """Tile evenly spaced frames of one gait cycle into a single image.

    The frames are sampled across the cycle rather than taken consecutively, so the strip
    reads as a stride: initial contact, loading, mid-stance, terminal stance, toe-off,
    swing.
    """
    import imageio.v2 as imageio

    indices = np.linspace(0, len(frames) - 1, columns).round().astype(int)
    strip = np.concatenate([frames[index] for index in indices], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, strip)
    print(f"[record_video] wrote {path} ({columns} postures across one gait cycle)")


def build_env(view: str):
    """Construct the environment with its viewer pointed at ``view``."""
    env_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "env_cfg_entry_point")
    env_cfg = resolve_presets(env_cfg, selected=tuple(args_cli.presets))
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    # A long episode so the clip is not interrupted by a timeout reset mid-stride.
    env_cfg.episode_length_s = 120.0
    configure_viewer(env_cfg, view=view, env_index=args_cli.env_index, resolution=tuple(args_cli.resolution))
    return gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array").unwrapped


def main() -> int:
    unknown = [view for view in args_cli.views if view not in VIEWS]
    if unknown:
        raise SystemExit(f"Unknown view(s) {unknown}. Available: {sorted(VIEWS)}")

    output_dir = Path(args_cli.output_dir or Path(args_cli.checkpoint).resolve().parent / "videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    stem = Path(args_cli.checkpoint).stem

    manifest = {"checkpoint": str(args_cli.checkpoint), "iteration": checkpoint.get("iteration", "?"), "clips": {}}

    # One environment per view: the Kit recorder resolves its render product once, and
    # rebuilding the scene is the reliable way to get a genuinely different camera.
    for view in args_cli.views:
        env = build_env(view)

        # Pin the paretic side; the play config still redraws it on every reset.
        side = -1.0 if args_cli.paretic_side == "left" else 1.0
        env.reference_gait.paretic_side[:] = side
        if env.cfg.events.reset_to_reference is not None:
            env.event_manager.get_term_cfg("reset_to_reference").params["randomize_paretic_side"] = False

        policy = ActorCritic(
            obs_dim=int(env.observation_space["policy"].shape[-1]),
            action_dim=int(env.action_space.shape[-1]),
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
        ).to(env.device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.eval()

        camera = TrackingCamera(env, view=view, env_index=args_cli.env_index)
        dt = float(env.step_dt)
        stride_duration = float(env.reference_gait.stride_duration_s)
        steps_per_cycle = max(int(round(stride_duration / dt)), 1)
        record_steps = int(round(args_cli.num_cycles * steps_per_cycle))

        print(f"[record_video] {view}: {record_steps} frames ({args_cli.num_cycles:g} cycles @ {dt * 1000:.0f} ms)")

        obs, _ = env.reset()
        obs = obs["policy"]
        frames: list[np.ndarray] = []

        for step in range(args_cli.warmup_steps + record_steps):
            with torch.no_grad():
                actions = policy.actor(obs)
            obs, _, _, _, _ = env.step(actions)
            obs = obs["policy"]

            camera.update()
            if step < args_cli.warmup_steps:
                continue

            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))

        frames = drop_warmup_frames(frames)
        if not frames:
            print(f"[record_video] {view}: renderer returned no frames; is --enable_cameras honoured on this backend?")
            env.close()
            continue

        video_path = output_dir / f"{stem}_{view}.mp4"
        write_video(frames, video_path, fps=1.0 / dt)
        manifest["clips"][view] = {"path": str(video_path), "frames": len(frames), "fps": 1.0 / dt}

        if args_cli.filmstrip:
            strip_path = output_dir / f"{stem}_{view}_filmstrip.png"
            write_filmstrip(frames[:steps_per_cycle], strip_path, args_cli.filmstrip_frames)
            manifest["clips"][view]["filmstrip"] = str(strip_path)

        env.close()

    (output_dir / "video_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[record_video] manifest -> {output_dir / 'video_manifest.json'}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
