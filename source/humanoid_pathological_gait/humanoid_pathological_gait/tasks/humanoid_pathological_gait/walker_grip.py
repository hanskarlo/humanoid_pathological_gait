# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The implicit grip: each hand tied to the handle by a PhysX D6 joint drive (forward plan Phase 4a).

The explicit grip (``mdp/walker.py``) applies ``F = K (p_grip - p_hand) + C (v_grip - v_hand)`` as
an external force each physics step. On the H1's light forearm (~0.2 kg effective at the grip) that
is unstable much above K = 5 kN/m at a 5 ms step
(``research_log/2026-10-07-walker-retest-no-go-...md``), and at 5 kN/m the hands sit a median
2.4 cm off the handle (``research_log/2026-10-22-the-walker-candidate-is-validated-...md``).

This spawner authors the same spring as a **joint drive**, which the solver integrates
implicitly, so it can be stiff:

* one generic ``UsdPhysics.Joint`` (a PhysX D6) per hand, from the elbow link's grip point
  (``hand_offset_body``) to the grip point on the handle (``+-half_grip_width`` along the bar,
  ``grip_height_offset`` up, in the walker root frame, as the explicit grip has it);
* the three translations **driven** (force-type drives, stiffness K, damping C, target 0), the
  three rotations **free**: the wrist free to rotate, as in the explicit model;
* ``excludeFromArticulation``: the joint ties two articulations, so it is a maximal-coordinate
  constraint, not an articulation joint.

The joints are authored once under ``env_0``. Cloning copies the whole ``env_0`` subtree, and
both copies remap targets inside it to each environment's own bodies: ``Sdf.CopySpec`` for the
USD, and the PhysX replicator's rename for the physics. **That remapping is the assumption to
check** when this first runs: every environment's joints must target its own robot and walker.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

import torch

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass

from . import walker_coupling as wc

TRANSLATION_AXES = ("transX", "transY", "transZ")


def _find_body(stage, root_path: str, name: str):
    """The unique prim named ``name`` under ``root_path``."""
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"grip joints: no prim at {root_path}; spawn the robot and the walker first")
    found = [p for p in Usd.PrimRange(root) if p.GetName() == name]
    if len(found) != 1:
        raise RuntimeError(f"grip joints: expected one '{name}' under {root_path}, found {len(found)}")
    return found[0]


def _world_quat_xyzw(prim) -> torch.Tensor:
    """The prim's world orientation as a ``(1, 4)`` ``(x, y, z, w)`` tensor. Refuses scaled frames."""
    from pxr import Gf, Usd, UsdGeom

    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    scale = Gf.Vec3d(*(Gf.Vec3d(matrix.GetRow3(i)).GetLength() for i in range(3)))
    if any(abs(s - 1.0) > 1e-6 for s in scale):
        raise RuntimeError(f"grip joints: {prim.GetPath()} has a scaled frame {tuple(scale)}; localPos would be scaled")
    q = matrix.ExtractRotationQuat()
    im = q.GetImaginary()
    return torch.tensor([[im[0], im[1], im[2], q.GetReal()]], dtype=torch.float64)


@clone
def spawn_grip_joints(prim_path: str, cfg: GripJointsCfg, translation=None, orientation=None, **kwargs):
    """Author the two grip joints under ``prim_path`` (an Xform in the environment's namespace)."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    del translation, orientation, kwargs
    stage = get_current_stage()
    env_root = prim_path.rsplit("/", 1)[0]
    robot_root, walker_root = f"{env_root}/{cfg.robot_prim_name}", f"{env_root}/{cfg.walker_prim_name}"
    hands = [_find_body(stage, robot_root, name) for name in cfg.hand_body_names]
    handle = _find_body(stage, walker_root, cfg.handle_body_name)

    d, h = cfg.half_grip_width, cfg.grip_height_offset
    offsets_root = torch.tensor([[0.0, d, h], [0.0, -d, h]], dtype=torch.float64)
    local_on_handle = wc.grip_offset_in_body_frame(
        _world_quat_xyzw(handle), _world_quat_xyzw(stage.GetPrimAtPath(walker_root)), offsets_root
    )

    xform = UsdGeom.Xform.Define(stage, prim_path)
    for i, (side, hand) in enumerate(zip(("left", "right"), hands)):
        _world_quat_xyzw(hand)  # refuses a scaled hand frame
        joint = UsdPhysics.Joint.Define(stage, f"{prim_path}/{side}_grip")
        joint.CreateBody0Rel().SetTargets([hand.GetPath()])
        joint.CreateBody1Rel().SetTargets([handle.GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*cfg.hand_offset_body))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*local_on_handle[i].tolist()))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateCollisionEnabledAttr(False)
        for axis in TRANSLATION_AXES:
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
            drive.CreateTypeAttr(UsdPhysics.Tokens.force)
            drive.CreateStiffnessAttr(float(cfg.stiffness))
            drive.CreateDampingAttr(float(cfg.damping))
            drive.CreateTargetPositionAttr(0.0)
        # Rotations are left free (no LimitAPI): a generic joint's axes are free unless limited.
        joint.GetPrim().CreateAttribute("grip:model", Sdf.ValueTypeNames.String).Set("implicit D6 drive")
    return xform.GetPrim()


@configclass
class GripJointsCfg(SpawnerCfg):
    """The implicit grip's two D6 joints. Geometry defaults match :class:`mdp.WalkerCouplingActionCfg`."""

    func: Callable = spawn_grip_joints
    stiffness: float = MISSING
    """N/m, per translational axis (isotropic)."""
    damping: float = MISSING
    """N s/m, per translational axis."""
    robot_prim_name: str = "Robot"
    walker_prim_name: str = "Walker"
    hand_body_names: tuple[str, str] = ("left_elbow_link", "right_elbow_link")
    handle_body_name: str = "handle"
    hand_offset_body: tuple[float, float, float] = (0.318, 0.0, -0.032)
    half_grip_width: float = 0.179
    grip_height_offset: float = 0.031
