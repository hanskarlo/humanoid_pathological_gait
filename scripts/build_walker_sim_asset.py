# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the simulation variant of the Smart Walker asset: primitive colliders, nothing else changed.

Pure ``pxr``, no Kit and no GPU. Run it with this extension's environment::

    .venv/bin/python scripts/build_walker_sim_asset.py

It reads ``data/walker/smart_walker.usda`` (the faithful reconstruction of the converter's output)
and writes two layers beside it:

* ``smart_walker_sim_overrides.usda`` -- the overrides, authored by this script;
* ``smart_walker_sim.usda`` -- the root to spawn: overrides sublayered over the faithful asset.

What changes, and why (``docs/walker_evaluation_plan.md`` §2.1-2.2):

* **Every mesh collider is disabled.** The converter produced 260 convex-decomposition mesh
  colliders (wheels, steering housings, the Ranger chassis). Cloned 14,720 times that is a
  performance and contact-stability risk, and none of that geometry is load-bearing in an
  idealised velocity-tracking base. The meshes live inside instanced prims, and USD does not
  accept overrides on instance proxies, so the instance roots are un-instanced first.
* **Four frictionless spheres replace the wheels**, one per wheel link at the wheel centre,
  with the radius measured from the wheel mesh (0.10 m). They carry the vertical load through
  ground contact. Being frictionless, they leave the planar motion entirely to the admittance
  command, which is what "ideal velocity tracking" means.
* **One box replaces the chassis**, sized from the chassis meshes' bounding box, so the H1's
  feet still collide with the walker.
* The handle, pillar and F/T-sensor primitives the converter authored (boxes and a cylinder)
  are kept as they are, and so is every mass, joint and ``ft_sensor_joint``.

The script refuses to write anything if the faithful asset's total mass is not 62.3 kg, or if the
geometry it measures differs from what the sizes below assume.
"""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

WALKER_DIR = (
    Path(__file__).resolve().parents[1]
    / "source/humanoid_pathological_gait/humanoid_pathological_gait/tasks/humanoid_pathological_gait/data/walker"
)
FAITHFUL = WALKER_DIR / "smart_walker.usda"
OVERRIDES = WALKER_DIR / "smart_walker_sim_overrides.usda"
ROOT = WALKER_DIR / "smart_walker_sim.usda"

BASE = "/smart_walker/Geometry/base_link"
WHEELS = ("fr", "fl", "rl", "rr")
EXPECTED_MASS_KG = 62.3
WHEEL_RADIUS_M = 0.10
"""Half the wheel mesh's vertical extent (0.61-0.811 m). The URDF says 0.09; the mesh is what
touches the ground in the faithful asset, so the collider matches the mesh."""


def world_bounds(stage: Usd.Stage, path: str) -> Gf.Range3d:
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    return cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()


def world_translation(stage: Usd.Stage, path: str) -> Gf.Vec3d:
    return UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0).ExtractTranslation()


def main() -> int:
    stage = Usd.Stage.Open(str(FAITHFUL))

    total_mass = sum(
        UsdPhysics.MassAPI(p).GetMassAttr().Get() or 0.0 for p in stage.Traverse() if p.HasAPI(UsdPhysics.MassAPI)
    )
    if abs(total_mass - EXPECTED_MASS_KG) > 0.05:
        raise SystemExit(f"faithful asset mass is {total_mass:.3f} kg, expected {EXPECTED_MASS_KG}; refusing to build")

    wheel_range = world_bounds(stage, f"{BASE}/fr_steering_wheel_link/fr_wheel_link")
    measured_radius = (wheel_range.GetMax()[2] - wheel_range.GetMin()[2]) / 2.0
    if abs(measured_radius - WHEEL_RADIUS_M) > 0.005:
        raise SystemExit(f"wheel mesh radius measures {measured_radius:.4f} m, expected {WHEEL_RADIUS_M}")

    chassis = world_bounds(stage, f"{BASE}/ranger_base")
    base_origin = world_translation(stage, BASE)
    chassis_center_local = (chassis.GetMin() + chassis.GetMax()) / 2.0 - base_origin
    chassis_size = chassis.GetMax() - chassis.GetMin()

    # Author the overrides into their own layer, sublayered strongest over the faithful asset.
    # Rebuilt from scratch on every run, so the output never carries a previous run's edits.
    for stale in (OVERRIDES, ROOT):
        stale.unlink(missing_ok=True)
    overrides = Sdf.Layer.CreateNew(str(OVERRIDES))
    root_layer = stage.GetRootLayer()
    root_layer.subLayerPaths.insert(0, OVERRIDES.name)
    stage.SetEditTarget(Usd.EditTarget(overrides))

    # 1. Un-instance every instance root, then disable every mesh collider underneath.
    # Un-instancing invalidates the proxies under it, and instances can nest, so collect the
    # instance roots visible on each pass and repeat until none are left.
    while True:
        roots = [prim.GetPath() for prim in stage.Traverse() if prim.IsInstance()]
        if not roots:
            break
        for path in roots:
            stage.OverridePrim(path).SetInstanceable(False)
    disabled = 0
    primitives = (UsdGeom.Cube, UsdGeom.Cylinder, UsdGeom.Sphere, UsdGeom.Capsule)
    for prim in stage.Traverse():
        # Anything that is not a primitive shape: the meshes, and one Xform the converter also
        # tagged with CollisionAPI. Only the converter's own handle/pillar/sensor primitives survive.
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not any(prim.IsA(kind) for kind in primitives):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            disabled += 1

    # 2. A frictionless physics material for the wheel spheres.
    material_path = Sdf.Path("/smart_walker/SimMaterials/frictionless")
    material = UsdShade.Material.Define(stage, material_path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(0.0)
    physics_material.CreateDynamicFrictionAttr(0.0)
    physics_material.CreateRestitutionAttr(0.0)

    # 3. Wheel spheres at the wheel-link origins (the wheel centres).
    for wheel in WHEELS:
        sphere = UsdGeom.Sphere.Define(stage, f"{BASE}/{wheel}_steering_wheel_link/{wheel}_wheel_link/sim_collider")
        sphere.CreateRadiusAttr(WHEEL_RADIUS_M)
        sphere.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )

    # 4. One chassis box, in base_link's frame (base_link carries no rotation in this asset).
    box = UsdGeom.Cube.Define(stage, f"{BASE}/sim_chassis_collider")
    box.CreateSizeAttr(1.0)
    box.CreatePurposeAttr(UsdGeom.Tokens.guide)
    box.AddTranslateOp().Set(Gf.Vec3d(*chassis_center_local))
    box.AddScaleOp().Set(Gf.Vec3f(*chassis_size))
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())

    overrides.Save()
    root_layer.subLayerPaths.remove(OVERRIDES.name)  # never save the edit into the faithful layer

    sim_root = Sdf.Layer.CreateNew(str(ROOT))
    sim_root.defaultPrim = "smart_walker"
    sim_root.subLayerPaths.append(OVERRIDES.name)
    sim_root.subLayerPaths.append(FAITHFUL.name)
    sim_root.documentation = (
        "Simulation variant of the Smart Walker: primitive colliders over the faithful asset. "
        "Generated by humanoid_pathological_gait/scripts/build_walker_sim_asset.py; do not edit by hand."
    )
    sim_root.Save()

    print(f"total mass {total_mass:.3f} kg (unchanged)")
    print(f"disabled {disabled} non-primitive colliders")
    print(f"wheel spheres r = {WHEEL_RADIUS_M} m at {', '.join(WHEELS)}")
    print(f"chassis box centre (base_link frame) {tuple(round(x, 4) for x in chassis_center_local)}")
    print(f"chassis box size {tuple(round(x, 4) for x in chassis_size)}")
    print(f"wrote {OVERRIDES.name} and {ROOT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
