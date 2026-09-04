# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A camera that follows one walking robot, plus the canonical views for gait figures.

Isaac Lab's ``--video`` path records the Kit perspective camera, and on the PhysX
backend that camera is positioned once, when the recorder is constructed, and never
moved again (``IsaacsimKitPerspectiveVideo.render_rgb_array`` calls
``set_camera_view`` only on its first frame). A robot walking at 0.5 m/s leaves the
frame within a few seconds, so a fixed camera is fine for a smoke test and useless
for a supplementary video. :class:`TrackingCamera` re-aims the same camera prim every
frame instead.

``ViewportCameraController`` would do this via ``origin_type="asset_root"``, but the
environment only constructs one when a Kit GUI or a visualizer is live -- which is
exactly not the case in the headless runs that render videos. This calls the viewport
API directly, so it works headless, and falls back to the controller when one exists.

The named views are the ones gait analysis is conventionally read in: **sagittal** for
joint flexion and foot clearance, **frontal** for mediolateral sway and the margin of
stability, **oblique** for a general-purpose figure.
"""

from __future__ import annotations

import numpy as np

#: Camera offsets from the tracked root, in metres, as ``(eye, lookat)`` in the world
#: frame. The robot walks along +x, so a sagittal view sits out along -y.
VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "sagittal": ((0.0, -3.2, 0.95), (0.0, 0.0, 0.85)),
    "frontal": ((4.0, 0.0, 1.1), (0.0, 0.0, 0.85)),
    "oblique": ((2.6, -2.8, 1.6), (0.0, 0.0, 0.80)),
    "wide": ((5.0, -5.5, 3.0), (0.0, 0.0, 0.70)),
}


class TrackingCamera:
    """Keeps the Kit perspective camera at a fixed offset from one environment's robot.

    Args:
        env: The unwrapped environment.
        view: A key of :data:`VIEWS`, or ``None`` when passing explicit offsets.
        env_index: Which environment to follow.
        eye_offset / lookat_offset: Explicit offsets, overriding ``view``.
        smoothing: Exponential smoothing on the tracked position, in ``[0, 1)``. The
            pelvis oscillates vertically and laterally through every stride; tracking it
            rigidly transfers that oscillation to the whole frame and makes the video
            hard to watch. Smoothing lets the camera follow the path rather than the gait.
    """

    def __init__(
        self,
        env,
        view: str | None = "sagittal",
        env_index: int = 0,
        eye_offset: tuple[float, float, float] | None = None,
        lookat_offset: tuple[float, float, float] | None = None,
        smoothing: float = 0.9,
    ):
        if view is not None and view not in VIEWS and (eye_offset is None or lookat_offset is None):
            raise ValueError(f"Unknown camera view '{view}'. Available: {sorted(VIEWS)}")

        default_eye, default_lookat = VIEWS.get(view or "sagittal")
        self._eye_offset = np.asarray(eye_offset if eye_offset is not None else default_eye, dtype=float)
        self._lookat_offset = np.asarray(lookat_offset if lookat_offset is not None else default_lookat, dtype=float)

        self._env = env
        self._env_index = env_index
        self._smoothing = float(np.clip(smoothing, 0.0, 0.99))
        self._tracked: np.ndarray | None = None
        self._set_camera_view = self._resolve_setter()

    def _resolve_setter(self):
        """Return a ``(eye, target) -> None`` callable for the recorded camera prim."""
        prim_path = getattr(self._env.cfg.viewer, "cam_prim_path", "/OmniverseKit_Persp")
        try:
            from isaacsim.core.rendering_manager import ViewportManager

            def setter(eye, target):
                ViewportManager.set_camera_view(prim_path, eye=list(eye), target=list(target))

            return setter
        except ImportError:
            # Newton / kitless installs: the simulation context still exposes the view.
            def setter(eye, target):
                self._env.sim.set_camera_view(eye=tuple(eye), target=tuple(target))

            return setter

    def _root_position(self) -> np.ndarray:
        """World-frame root position of the tracked robot."""
        robot = self._env.scene["robot"]
        return robot.data.root_link_pos_w.torch[self._env_index].detach().cpu().numpy().astype(float)

    def update(self) -> None:
        """Re-aim the camera at the tracked robot. Call once per rendered frame."""
        position = self._root_position()
        if self._tracked is None:
            self._tracked = position
        else:
            self._tracked = self._smoothing * self._tracked + (1.0 - self._smoothing) * position

        # Height is taken from the ground rather than the pelvis: following the vertical
        # bob of the centre of mass makes the horizon rock through every stride.
        anchor = np.array([self._tracked[0], self._tracked[1], self._env.scene.env_origins[self._env_index, 2].item()])
        self._set_camera_view(anchor + self._eye_offset, anchor + self._lookat_offset)


def configure_viewer(env_cfg, view: str = "sagittal", env_index: int = 0, resolution: tuple[int, int] = (1920, 1080)):
    """Point a config's viewer and video recorder at ``view`` before the environment is built.

    The recorder copies ``viewer.eye``/``viewer.lookat`` at construction, so setting them
    here is what gives the very first recorded frame a sensible pose;
    :class:`TrackingCamera` takes over from the second frame on.
    """
    eye_offset, lookat_offset = VIEWS[view]
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = env_index
    env_cfg.viewer.eye = tuple(eye_offset)
    env_cfg.viewer.lookat = tuple(lookat_offset)
    env_cfg.viewer.resolution = resolution
    if getattr(env_cfg, "video_recorder", None) is not None:
        env_cfg.video_recorder.eye = tuple(eye_offset)
        env_cfg.video_recorder.lookat = tuple(lookat_offset)
        env_cfg.video_recorder.window_width = resolution[0]
        env_cfg.video_recorder.window_height = resolution[1]
    return env_cfg
