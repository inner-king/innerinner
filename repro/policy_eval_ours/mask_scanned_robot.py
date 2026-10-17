# SPDX-License-Identifier: Apache-2.0
"""Hide the Franka that the DROID desk scan droid_v1 captured, in a new background layer.

The authors' mesh background droid_v1 (a Scaniverse phone scan of DROID Station 1) contains the real robot, and the
DROID scenes register the scan so that the simulated Franka's base sits on the scanned one. Once the simulated arm
moves, the scanned arm stays where it was and is drawn into every exterior-camera frame the policy receives: a second
robot the real camera never sees (repro/README.md, chapter 23). The background has no collision geometry, so it only
changes what the policy sees.

This writes `<name>_norobot.usd` next to the background: a layer that references the same USDZ and lists the faces
of the scanned arm in the mesh's `holeIndices`, which renderers skip. The scan itself is not touched (AGENTS.md rule
1) and no scene changes unless it is pointed at the new layer.

Finding the arm: the scan's mesh is many disconnected patches, so its own connectivity does not hold the arm
together. Instead the vertices in a box around the scene's robot base, from `--min-height` above the base (which
leaves the cart's top plate) up to 1 m, are linked when closer than `--link` and split into connected groups; the
arm is the group holding the vertices just above the base. The scan's arm stands in a different pose from the
joints its background file gives the scene's robot (it rises to 0.76 m, those joints to 0.58 m), so the robot's
kinematics would not locate it.

Runs anywhere usd-core and scipy are importable (the 3dgrut env here):
    python repro/policy_eval_ours/mask_scanned_robot.py \\
        --background assets/backgrounds/mesh_backgrounds/droid_v1.usd \\
        --scene assets/scenes/DROID/droid_desk_put_marker_in_cup/droid_desk_put_marker_in_cup_scene_state_latest.json
"""
import argparse
import json
import os

import numpy as np
from pxr import Sdf, Usd, UsdGeom, Vt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--background", required=True, help="the background's .usd")
    ap.add_argument("--scene", required=True, help="a scene that uses it: its background and robot poses are used")
    ap.add_argument("--min-height", type=float, default=0.03, help="ignore everything this close above the base (m)")
    ap.add_argument("--link", type=float, default=0.025, help="vertices closer than this belong together (m)")
    ap.add_argument("--out", default=None, help="default: <background>_norobot.usd")
    args = ap.parse_args()
    out = args.out or f"{os.path.splitext(args.background)[0]}_norobot.usd"

    reg = json.load(open(args.scene))["state"]["registry"]["object_registry"]
    bg = next(v["root_link"] for k, v in reg.items() if k.startswith("mesh_background"))
    base = np.array(reg["robot0"]["root_link"]["pos"])
    to_robot = Rotation.from_quat(reg["robot0"]["root_link"]["ori"]).inv()

    stage = Usd.Stage.Open(args.background)
    mesh_prim = next(p for p in stage.Traverse() if p.GetTypeName() == "Mesh")
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    to_root = np.array(UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    world = Rotation.from_quat(bg["ori"]).apply((np.c_[pts, np.ones(len(pts))] @ to_root.T)[:, :3]) + np.array(bg["pos"])
    rel = to_robot.apply(world - base)  # in the robot's base frame: x forward, z up

    box = np.flatnonzero((rel[:, 0] > -0.5) & (rel[:, 0] < 0.6) & (np.abs(rel[:, 1]) < 0.35)
                         & (rel[:, 2] > args.min_height) & (rel[:, 2] < 1.0))
    pairs = cKDTree(rel[box]).query_pairs(args.link, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(box), len(box)))
    _, group = connected_components(graph, directed=False)
    seeds = (np.hypot(rel[box, 0], rel[box, 1]) < 0.08) & (rel[box, 2] > 0.1) & (rel[box, 2] < 0.3)
    if not seeds.any():
        raise SystemExit("no scan vertices just above the robot base: is the scene's robot on the scanned one?")
    arm = np.zeros(len(world), dtype=bool)
    arm[box[group == np.bincount(group[seeds]).argmax()]] = True

    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    hide = np.zeros(len(counts), dtype=bool)
    np.logical_or.at(hide, np.repeat(np.arange(len(counts)), counts), arm[indices])
    holes = np.flatnonzero(hide)

    layer = Sdf.Layer.CreateNew(out, args={"format": "usda"})
    out_stage = Usd.Stage.Open(layer)
    out_stage.SetMetadata("metersPerUnit", UsdGeom.GetStageMetersPerUnit(stage))
    out_stage.SetMetadata("upAxis", UsdGeom.GetStageUpAxis(stage))
    # The original wrapper's structure, so the prim paths a scene sees stay the same.
    src_root = stage.GetDefaultPrim()
    root = out_stage.DefinePrim(src_root.GetPath(), "Xform")
    out_stage.SetDefaultPrim(root)
    for prim in src_root.GetChildren():
        refs = prim.GetMetadata("references")
        child = out_stage.DefinePrim(prim.GetPath(), prim.GetTypeName())
        for ref in (refs.prependedItems if refs else []):
            child.GetReferences().AddReference(ref.assetPath, ref.primPath)
    UsdGeom.Mesh(out_stage.OverridePrim(mesh_prim.GetPath())).CreateHoleIndicesAttr().Set(
        Vt.IntArray([int(i) for i in holes]))
    lo, hi = rel[arm].min(0), rel[arm].max(0)
    layer.customLayerData = {"comment": f"{os.path.basename(args.background)} with its scanned Franka hidden "
                                        f"(repro/policy_eval_ours/mask_scanned_robot.py): {len(holes)} faces"}
    layer.Save()
    print(f"[mask] {out}: {arm.sum()} vertices and {len(holes)} of {len(counts)} faces hidden; in the robot frame "
          f"they span x {lo[0]:.2f}..{hi[0]:.2f}, y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f} m")


if __name__ == "__main__":
    main()
