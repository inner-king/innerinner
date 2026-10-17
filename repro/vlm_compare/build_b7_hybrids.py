# SPDX-License-Identifier: Apache-2.0
"""Scene states for B7's scene image with the PutMarkerInCup scenes' object meshes and poses crossed.

replay_b7_cross.py found qwen3-vl putting the scissors in the cup more often on put_marker_in_cup_q35's
scene image than on put_marker_in_cup_v2's, whatever the object list (repro/README.md, chapter 16). That
image (s14_og/auto_generation/scene_000.png) is B6's render of Pipeline A's meshes at the poses B6's first
randomized iteration settled to, which B6 saved beside it as scene_000.json. The two scenes differ in both,
so this writes four scene states for render_b7_scenes.py, one per image:

  v2                   v2's scene_000.json as saved
  q35                  q35's scene_000.json as saved
  q35_meshes_v2_poses  q35's three meshes lying where v2's lie in v2's image (v2's file, meshes swapped)
  v2_meshes_q35_poses  v2's three meshes lying where q35's lie in q35's image

Objects are matched by kind (marker, cup, scissors), since their iter_ ids differ between the scenes.

A root pose does not carry over from one reconstruction of an object to another: each mesh has its own
local frame. q35's scissors lie 47 degrees from v2's in their frames, and q35's marker points the other
way along its axis, so copying root_link.ori would turn the scissors and put the marker's cap at the
wrong end. Each q35 mesh is therefore registered onto the v2 mesh of its kind in their local frames
(rigid, on surface samples: random rotations refined by ICP, and among the fits within 5% of the best
fit's distance, the one whose texture colours agree best, which is what settles the marker's ends and the
cup's yaw), and a mesh takes the other scene's root pose composed with that registration. It is then
raised or lowered so its lowest point sits where the replaced mesh's did, on the floor.

Writes <out_dir>/<name>.json and <out_dir>/registration.json; nothing under Data/.
Usage: python repro/vlm_compare/build_b7_hybrids.py <out_dir> [--data-root DIR]   (about 11 minutes, CPU)
"""
import argparse
import copy
import glob
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
SCENES = ("put_marker_in_cup_v2", "put_marker_in_cup_q35")
SCENE_STATE = Path("s14_og/auto_generation/scene_000.json")
ASSETS = REPO / "deps/BEHAVIOR-1K/datasets/real2sim-assets/objects"
KINDS = ("marker", "cup", "scissors")
# Fits within this fraction of the best fit's distance count as equally good geometrically; texture
# colour picks among them. The marker is nearly symmetric end to end and the cup about its axis.
NEAR_TIE = 0.05


def kind_of(category):
    return next(k for k in KINDS if k in category)


def scene_objects(state):
    """kind -> (object id, init args, root_link) for the three props of a saved scene state."""
    out = {}
    for name, info in state["objects_info"]["init_info"].items():
        args = info["args"]
        if info["class_name"] == "DatasetObject":
            out[kind_of(args["category"])] = (name, args, state["state"]["registry"]["object_registry"][name]["root_link"])
    assert sorted(out) == sorted(KINDS), out.keys()
    return out


def load_mesh(args):
    (obj,) = glob.glob(str(ASSETS / args["category"] / args["model"] / "urdf/shape/visual/*.obj"))
    return trimesh.load(obj, process=False)


def surface(mesh, n, seed):
    pts, _, col = trimesh.sample.sample_surface(mesh, n, sample_color=True, seed=seed)
    return np.asarray(pts, float), np.asarray(col, float)[:, :3] / 255.0


def kabsch(a, b):
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    rot = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return rot, cb - rot @ ca


def icp(src, tree, tgt, rot, t, iters=40, keep=0.9):
    """Trimmed point-to-point ICP of src onto tgt (tree over tgt), from (rot, t)."""
    for _ in range(iters):
        d, idx = tree.query(src @ rot.T + t)
        inl = d <= np.quantile(d, keep)
        rot, t = kabsch(src[inl], tgt[idx[inl]])
    return rot, t


def fit(src, scol, tgt, tcol, rot, t):
    """Symmetric mean surface distance (m) and mean colour difference (0-1) of src moved onto tgt."""
    moved = src @ rot.T + t
    d1, i1 = cKDTree(tgt).query(moved)
    d2, i2 = cKDTree(moved).query(tgt)
    return float((d1.mean() + d2.mean()) / 2), float((np.abs(scol - tcol[i1]).mean() + np.abs(tcol - scol[i2]).mean()) / 2)


def angle_deg(r1, r2):
    return float(np.degrees(Rotation.from_matrix(r1 @ r2.T).magnitude()))


def register(src_mesh, tgt_mesh, seed=0):
    """Rigid (R, t) taking src_mesh's local frame onto tgt_mesh's (x_tgt = R x_src + t), and the evidence."""
    src, scol = surface(src_mesh, 20000, seed)
    tgt, tcol = surface(tgt_mesh, 20000, seed + 1)
    sc, tc = src.mean(0), tgt.mean(0)
    src0, tgt0 = src - sc, tgt - tc
    tree = cKDTree(tgt0)
    rng = np.random.default_rng(seed)
    probe = src0[rng.choice(len(src0), 400, replace=False)]
    rots = Rotation.random(30000, random_state=seed).as_matrix()
    coarse = np.array([tree.query(probe @ r.T)[0].mean() for r in rots])
    sub = src0[rng.choice(len(src0), 4000, replace=False)]
    cands = []
    for i in np.argsort(coarse)[:60]:
        rot, t = icp(sub, tree, tgt0, rots[i], np.zeros(3))
        cands.append((*fit(src0, scol, tgt0, tcol, rot, t), rot, t))
    # Geometric near-ties differ along a near-symmetry; small random turns about the colour-best of them
    # let colour settle that direction more finely than the 30k-rotation grid does.
    best_d = min(c[0] for c in cands)
    d0, c0, r0, t0 = min((c for c in cands if c[0] <= best_d * (1 + NEAR_TIE)), key=lambda c: c[1])
    turns = Rotation.from_rotvec(Rotation.random(3000, random_state=seed + 2).as_rotvec()
                                 * rng.uniform(0, np.radians(15), 3000)[:, None]).as_matrix()
    for turn in turns:
        rot = turn @ r0
        cands.append((*fit(src0, scol, tgt0, tcol, rot, turn @ t0), rot, turn @ t0))
    best_d = min(c[0] for c in cands)
    d, c, rot, t = min((x for x in cands if x[0] <= best_d * (1 + NEAR_TIE)), key=lambda x: x[1])
    # The best fit that is not a near-copy of the chosen one, to show how decisive the choice was.
    other = min((x for x in cands if angle_deg(x[2], rot) > 30), key=lambda x: x[0] + 0.01 * x[1], default=None)
    raw_d, raw_c = fit(src0, scol, tgt0, tcol, np.eye(3), np.zeros(3))
    rv = Rotation.from_matrix(rot).as_rotvec()
    return rot, t + tc - rot @ sc, {
        "rotation_deg": round(float(np.degrees(np.linalg.norm(rv))), 1),
        "rotation_axis": np.round(rv / (np.linalg.norm(rv) + 1e-12), 3).tolist(),
        "surface_distance_mm": round(d * 1000, 2), "colour_difference": round(c, 3),
        "next_best_other_fit": None if other is None else {
            "angle_from_chosen_deg": round(angle_deg(other[2], rot), 1),
            "surface_distance_mm": round(other[0] * 1000, 2), "colour_difference": round(other[1], 3)},
        "root_frames_as_is": {"surface_distance_mm": round(raw_d * 1000, 2), "colour_difference": round(raw_c, 3)},
    }


def pose_matrix(root):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(root["ori"]).as_matrix()  # OmniGibson and scipy are both xyzw
    m[:3, 3] = root["pos"]
    return m


def placement(mesh, m):
    """Where a mesh at world pose m lies: footprint centre, long-axis yaw, floor contact, extent."""
    v = mesh.vertices @ m[:3, :3].T + m[:3, 3]
    lo, hi = v.min(0), v.max(0)
    xy = v[:, :2] - v[:, :2].mean(0)
    w, e = np.linalg.eigh(np.cov(xy.T))
    ax = e[:, np.argmax(w)]
    return {"footprint_centre_m": np.round((lo[:2] + hi[:2]) / 2, 4).tolist(),
            "long_axis_yaw_deg": round(float(np.degrees(np.arctan2(ax[1], ax[0])) % 180), 1),
            "lowest_z_m": round(float(lo[2]), 4), "height_m": round(float(hi[2] - lo[2]), 4)}


def hybrid(pose_state, mesh_state, meshes, regs, pose_scene, mesh_scene):
    """pose_state with each prop's mesh replaced by mesh_state's of its kind, laid over the replaced one."""
    out = copy.deepcopy(pose_state)
    pose_objs, mesh_objs = scene_objects(pose_state), scene_objects(mesh_state)
    report = {}
    for kind in KINDS:
        name, _, root = pose_objs[kind]
        _, margs, _ = mesh_objs[kind]
        rot, t = regs[kind]  # q35 local -> v2 local
        reg = np.eye(4)
        reg[:3, :3], reg[:3, 3] = rot, t
        if mesh_scene == SCENES[1]:  # a q35 mesh over a v2 one
            local = reg
        else:  # a v2 mesh over a q35 one
            local = np.linalg.inv(reg)
        m = pose_matrix(root) @ local
        old = placement(meshes[pose_scene][kind], pose_matrix(root))
        m[2, 3] += old["lowest_z_m"] - placement(meshes[mesh_scene][kind], m)["lowest_z_m"]
        args = out["objects_info"]["init_info"][name]["args"]
        for key in ("category", "model", "dataset_name", "expected_file_hash", "scale"):
            args[key] = margs[key]
        link = out["state"]["registry"]["object_registry"][name]["root_link"]
        link["pos"] = m[:3, 3].tolist()
        link["ori"] = Rotation.from_matrix(m[:3, :3]).as_quat().tolist()
        link["lin_vel"], link["ang_vel"] = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        report[kind] = {"id": name, "mesh": f"{margs['category']}/{margs['model']}", "replaces": old,
                        "placed": placement(meshes[mesh_scene][kind], m)}
    return out, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--data-root", default=str(REPO / "Data"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    states = {s: json.loads((Path(args.data_root) / s / SCENE_STATE).read_text()) for s in SCENES}
    objs = {s: scene_objects(states[s]) for s in SCENES}
    meshes = {s: {k: load_mesh(objs[s][k][1]) for k in KINDS} for s in SCENES}
    v2, q35 = SCENES
    regs, evidence = {}, {}
    for kind in KINDS:
        rot, t, ev = register(meshes[q35][kind], meshes[v2][kind])
        regs[kind] = (rot, t)
        evidence[kind] = {"q35_mesh": "/".join(objs[q35][kind][1][k] for k in ("category", "model")),
                          "v2_mesh": "/".join(objs[v2][kind][1][k] for k in ("category", "model")), **ev}
        print(kind, json.dumps(evidence[kind]), flush=True)
    scenes = {"v2": (states[v2], {}), "q35": (states[q35], {})}
    scenes["q35_meshes_v2_poses"] = hybrid(states[v2], states[q35], meshes, regs, v2, q35)
    scenes["v2_meshes_q35_poses"] = hybrid(states[q35], states[v2], meshes, regs, q35, v2)
    for name, (state, _) in scenes.items():
        (out / f"{name}.json").write_text(json.dumps(state, indent=4) + "\n")
    report = {"registration_q35_onto_v2": evidence,
              "placements": {name: rep for name, (_, rep) in scenes.items() if rep},
              "originals": {s: {k: {"id": objs[s][k][0], **placement(meshes[s][k], pose_matrix(objs[s][k][2]))}
                                for k in KINDS} for s in SCENES}}
    (out / "registration.json").write_text(json.dumps(report, indent=1) + "\n")
    for name, (_, rep) in scenes.items():
        for kind, r in rep.items():
            print(f"{name:20s} {kind:8s} {r['mesh']:36s} replaces {r['replaces']}\n{'':67s}placed {r['placed']}")


if __name__ == "__main__":
    main()
