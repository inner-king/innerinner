# SPDX-License-Identifier: Apache-2.0
"""Cut published fruit meshes by planes offline, for the runtime-cutting benchmarks (repro/README.md, section 15.1).

Two subcommands, in two environments, because no single env here has both a standalone ``pxr`` and the mesh tools:

``export`` (3dgrut env: standalone pxr) reads each fruit's visual mesh, its convex collision hulls, link name and
mass out of the scene's USD assets and writes one ``<object>.npz`` per fruit::

    ~/miniforge3/envs/3dgrut/bin/python -I repro/fruit_slicing/prepare_cut_pieces.py export \\
        --scene /home/inner/real2sim/SimFoundry/assets/scenes/DROID/droid_desk_serve_fruits/droid_desk_serve_fruits_scene_state_latest.json \\
        --out results/diagnostics/fruit_slicing/pieces

``cut`` (simfoundry env: trimesh, scikit-image, pymeshlab, coacd; no pxr) cuts every fruit by ``--n-planes`` planes and
writes, per cut, both halves with every collision variant the cooking benchmark adds (bench_cooking.py)::

    PYTHONPATH=$PWD ~/miniforge3/envs/simfoundry/bin/python repro/fruit_slicing/prepare_cut_pieces.py cut \\
        --out results/diagnostics/fruit_slicing/pieces

Per cut, each half carries:

- ``visual``: the fruit's full-resolution visual mesh sliced by ``trimesh.Trimesh.slice_plane`` and closed by a
  flat cap (close_rim). Triangles are cut, not assigned by centroid, so it is what a runtime cut would author.
- ``proxy``: the same plane applied to a simplified collision proxy of the whole fruit (rebuilt watertight and
  decimated once, offline, to ``--proxy-faces`` triangles: SDF cooking and V-HACD need a closed solid).
- ``clip``: the asset's own convex collision hulls, each intersected with the half-space. A convex polytope cut by a
  plane stays convex, so this decomposes a piece without running a decomposition at cut time.
- ``coacd<N>``: CoACD on the proxy half with ``max_convex_hull=N``, for N in ``--coacd`` (8 by default: CoACD's time
  hardly depends on N, and PhysX's own decomposition is measured at 4, 8 and 16 by bench_cooking.py).

The time each step takes is recorded too (``manifest.json``): a cut made at runtime would pay for it on the CPU
before PhysX cooks anything. The planes are random but seeded, and every one leaves each half at least
``--min-volume-frac`` of the fruit, so every cut makes a distinct mesh and PhysX's cooking cache never hits.
"""
import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

FRUITS = ("red_apple_zlgmzo_2", "yellow_banana_yvjcfy_3", "yellow_pear_uckunr_0", "orange_fruit_rofqtq_4")


# ------------------------------------------------------------------------------------------------------ export
def export(args):
    """Write each fruit's meshes, link name, mass and scene pose to ``<out>/<object>.npz`` (needs standalone pxr)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    scene = json.load(open(args.scene))
    init = scene["objects_info"]["init_info"]
    reg = scene["state"]["registry"]["object_registry"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.fruits:
        usd_path = Path(args.scene).parent / init[name]["args"]["usd_path"]
        stage = Usd.Stage.Open(str(usd_path))
        root = stage.GetDefaultPrim()
        link = next(c for c in root.GetChildren() if c.HasAPI(UsdPhysics.RigidBodyAPI))
        root_M = np.array(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        assert np.allclose(root_M, np.eye(4), atol=1e-6), f"{name}: non-identity root transform"
        vis, hulls = None, []
        for prim in Usd.PrimRange(link):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            # The pieces are authored in the fruit's frame, so every mesh is brought into it. The collision hulls
            # carry a per-hull scale op (their points are normalized); the visual mesh's transform is identity.
            M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            mesh = UsdGeom.Mesh(prim)
            pts = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
            pts = (np.column_stack([pts, np.ones(len(pts))]) @ M)[:, :3]  # USD matrices act on row vectors
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                hulls.append(pts)
            elif "/visuals/" in str(prim.GetPath()):
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
                assert (counts == 3).all(), "visual mesh is not triangulated"
                faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
                vis = (pts, faces)
        assert vis is not None and hulls, f"{name}: no visual mesh or no collision hulls"
        mass = link.GetAttribute("physics:mass").Get()
        rl = reg[name]["root_link"]
        np.savez_compressed(
            out / f"{name}.npz", V=vis[0], F=vis[1], hull_points=np.concatenate(hulls),
            hull_offsets=np.cumsum([0] + [len(h) for h in hulls]), mass=float(mass or 0.0),
            link_name=link.GetName(), usd_path=str(usd_path.resolve()), pos=np.asarray(rl["pos"]),
            ori=np.asarray(rl["ori"]))
        print(f"{name}: {len(vis[1])} visual triangles, {len(hulls)} collision hulls, link {link.GetName()}, "
              f"mass {mass} kg", flush=True)


# --------------------------------------------------------------------------------------------------------- cut
def close_rim(V, F):
    """Close the open rim a plane slice leaves: one triangle per rim edge, fanned to the centroid of its rim loop.

    Each rim edge gets exactly one cap triangle, wound against the side face it borders, so the result is closed
    whenever the slice was, and the fans' signed areas sum to each loop's area even where a loop is not convex or
    sits inside another (a plane through the apple proxy's stem dimple leaves an annulus). Triangulating the
    section polygons instead (shapely's constrained Delaunay) left holes: GEOS rejects a rim loop that grazes the
    surface as a self-touching sliver.

    Args:
        V (np.ndarray): (n, 3) vertices of the sliced mesh, welded
        F (np.ndarray): (m, 3) faces

    Returns:
        tuple: (V, F) with one centroid vertex per rim loop appended and the cap triangles appended
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    import trimesh

    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    rim = edges[trimesh.grouping.group_rows(np.sort(edges, axis=1), require_count=1)]
    if len(rim) == 0:
        return V, F
    _, label = connected_components(coo_matrix((np.ones(len(rim)), (rim[:, 0], rim[:, 1])), shape=(len(V),) * 2),
                                     directed=False)
    loops, loop_of_edge = np.unique(label[rim[:, 0]], return_inverse=True)
    centers = np.array([V[np.unique(rim[loop_of_edge == i])].mean(0) for i in range(len(loops))])
    cap = np.column_stack([rim[:, 1], rim[:, 0], len(V) + loop_of_edge])
    return np.concatenate([V, centers]), np.concatenate([F, cap])


def cut_half(mesh, origin, normal):
    """The part of ``mesh`` on the ``+normal`` side of the plane, closed by a flat cap.

    Returns:
        tuple: (trimesh.Trimesh half, dict of seconds spent slicing and capping)
    """
    import trimesh

    t0 = time.perf_counter()
    side = mesh.slice_plane(plane_origin=origin, plane_normal=normal, cap=False)
    side.merge_vertices()  # the slice gives every cut triangle its own crossing points; weld them into one rim
    t1 = time.perf_counter()
    V, F = close_rim(side.vertices, side.faces)
    t2 = time.perf_counter()
    return trimesh.Trimesh(V, F, process=False), {"slice_s": t1 - t0, "cap_s": t2 - t1}


def clip_hull(points, origin, normal):
    """Intersect a convex hull with the half-space on the ``+normal`` side.

    The result is the hull of the vertices on that side plus the points where the hull's edges cross the plane.

    Returns:
        tuple or None: (V, F) of the clipped hull, outward wound; None when no more than a sliver survives
    """
    import trimesh
    from scipy.spatial import ConvexHull, QhullError

    hull = ConvexHull(points)
    d = (points - origin) @ normal
    keep = points[d >= 0]
    edges = {tuple(sorted((s[i], s[(i + 1) % 3]))) for s in hull.simplices for i in range(3)}
    cross = [points[a] + (points[b] - points[a]) * (d[a] / (d[a] - d[b]))
             for a, b in edges if (d[a] >= 0) != (d[b] >= 0)]
    P = np.concatenate([keep, np.asarray(cross).reshape(-1, 3)])
    if len(P) < 4:
        return None
    try:
        clipped = trimesh.convex.convex_hull(P)
    except QhullError:  # flat: the plane only grazes this hull
        return None
    if clipped.volume < 1e-9:
        return None
    return clipped.vertices, clipped.faces


def make_proxy(mesh, n_faces, voxels=96):
    """Simplified, watertight collision proxy of the whole fruit.

    The visual meshes are closed but not manifold (trimesh reports edges shared by more than two faces), which
    SDF cooking and V-HACD do not accept as a solid. So the fruit is voxelized (``voxels`` cells along its longest
    side), filled, and turned back into a surface by marching cubes, which is closed and manifold by
    construction; quadric decimation then brings it to ``n_faces`` without changing its topology. (MeshLab's
    uniform resampling, tried first, returned over a thousand disconnected fragments on these meshes.)

    Args:
        mesh (trimesh.Trimesh): the fruit's visual mesh, vertices welded
        n_faces (int): triangle budget of the proxy
        voxels (int): voxel cells along the longest side

    Returns:
        tuple: (trimesh.Trimesh, dict of seconds spent and the voxel size used)
    """
    import pymeshlab
    import trimesh

    t0 = time.perf_counter()
    pitch = float(np.ptp(mesh.vertices, axis=0).max()) / voxels
    grid = mesh.voxelized(pitch).fill()
    surface = grid.marching_cubes
    surface.apply_transform(grid.transform)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=surface.vertices, face_matrix=surface.faces.astype(np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces, preservetopology=True, preservenormal=True,
                                                planarquadric=True)
    m = ms.current_mesh()
    proxy = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=True)
    trimesh.repair.fix_normals(proxy)
    return proxy, {"proxy_s": time.perf_counter() - t0, "voxel_m": pitch}


def _coacd_child(V, F, n_max, seed, preprocess, out):
    import coacd

    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(V, F), threshold=0.03, max_convex_hull=n_max, preprocess_mode=preprocess,
                            seed=seed)
    out.put([(np.asarray(v, np.float64), np.asarray(f, np.int64)) for v, f in parts])


def run_coacd(mesh, n_max, seed, timeout=300):
    """CoACD decomposition of a watertight mesh into at most ``n_max`` convex parts, in a forked child.

    Threshold 0.03 rather than CoACD's 0.05: at 0.05 an apple half stops at about 10 parts when allowed 16. 0.01
    took minutes per half. The rest are CoACD's defaults (resolution 2000, MCTS 150 iterations, depth 3).

    CoACD's own manifold preprocessing is always on. Without it, CoACD segfaulted on most apple halves (the apple
    proxy has tunnels: genus 1-9 after a cut) and ran past 15 minutes on one, while with it every half tried
    finished. Its time hardly depends on ``n_max`` (18-19 s for 4, 8 and 16 parts on one apple half). The child
    keeps a crash or a hang from ending the run: either is recorded and the half gets no CoACD hulls.

    Returns:
        tuple: (list of (V, F) per convex part, or None on failure; {s, exit_code})
    """
    import multiprocessing as mp
    import queue

    ctx = mp.get_context("fork")
    out = ctx.Queue()
    t0 = time.perf_counter()
    proc = ctx.Process(target=_coacd_child, args=(mesh.vertices, mesh.faces, n_max, seed, "on", out))
    proc.start()
    parts = None
    while parts is None and time.perf_counter() - t0 < timeout:
        try:
            parts = out.get(timeout=0.5)
        except queue.Empty:
            if not proc.is_alive():
                break
    seconds = time.perf_counter() - t0
    proc.join(5)
    if proc.is_alive():
        proc.kill()
        proc.join()
    info = {"s": seconds, "exit_code": "timeout" if parts is None and seconds >= timeout else proc.exitcode}
    if parts is None:
        print(f"    coacd (max {n_max}) failed: {info['exit_code']}", flush=True)
    return parts, info


def pack_parts(parts):
    """Ragged list of (V, F) meshes -> dict of concatenated arrays and offsets for npz (bench_cooking.unpack)."""
    parts = list(parts)
    return {"V": np.concatenate([v for v, _ in parts]) if parts else np.zeros((0, 3)),
            "VO": np.cumsum([0] + [len(v) for v, _ in parts]).astype(np.int64),
            "F": np.concatenate([f for _, f in parts]) if parts else np.zeros((0, 3), np.int64),
            "FO": np.cumsum([0] + [len(f) for _, f in parts]).astype(np.int64)}


def sample_plane(rng, mesh, min_frac):
    """A random plane through the fruit leaving each side at least ``min_frac`` of its volume (by vertex count of
    the proxy, which is close enough for choosing a plane)."""
    c = mesh.vertices.mean(0)
    for _ in range(1000):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        d = (mesh.vertices - c) @ n
        o = c + n * rng.uniform(np.quantile(d, 0.3), np.quantile(d, 0.7))
        side = ((mesh.vertices - o) @ n) > 0
        if min(side.mean(), 1 - side.mean()) >= min_frac:
            return o, n
    raise RuntimeError("no plane found")


def cut(args):
    """Cut every exported fruit by ``--n-planes`` planes; write ``<object>_cut<k>.npz`` and ``manifest.json``."""
    import trimesh

    out = Path(args.out)
    manifest = {"conditions": {"proxy_faces": args.proxy_faces, "coacd_max_hulls": args.coacd,
                               "n_planes": args.n_planes, "seed": args.seed, "trimesh": trimesh.__version__,
                               "cpu_count": os.cpu_count(), "loadavg_at_start": os.getloadavg()},
                "fruits": {}}
    for name in args.fruits:
        z = np.load(out / f"{name}.npz")
        V, F = z["V"], z["F"]
        hulls = [z["hull_points"][a:b] for a, b in zip(z["hull_offsets"][:-1], z["hull_offsets"][1:])]
        t0 = time.perf_counter()
        visual = trimesh.Trimesh(V, F, process=False)
        visual.merge_vertices()  # the USD mesh repeats vertices along UV seams; a section needs them welded
        t_weld = time.perf_counter() - t0
        proxy, proxy_info = make_proxy(visual, args.proxy_faces)
        assert proxy.is_watertight, f"{name}: proxy is not watertight"
        entry = {"visual_faces": int(len(F)), "visual_vertices_welded": int(len(visual.vertices)),
                 "weld_visual_s": t_weld, "proxy_faces": int(len(proxy.faces)), "proxy_volume_m3": float(proxy.volume),
                 "asset_hulls": len(hulls), "mass_kg": float(z["mass"]), "link_name": str(z["link_name"]),
                 **proxy_info, "cuts": []}
        print(f"{name}: visual {len(F)} tris, proxy {len(proxy.faces)} tris (watertight, "
              f"{proxy.volume * 1e6:.1f} cm^3, {proxy_info['proxy_s']:.2f} s), {len(hulls)} asset hulls", flush=True)
        rng = np.random.default_rng([args.seed, zlib.crc32(name.encode())])
        for k in range(args.n_planes):
            o, n = sample_plane(rng, proxy, args.min_volume_frac)
            rec = {"k": k, "origin": o.tolist(), "normal": n.tolist(), "halves": []}
            arrays = {"origin": o, "normal": n}
            for side, sgn in enumerate((1.0, -1.0)):
                hv, tv = cut_half(visual, o, sgn * n)
                hp, tp = cut_half(proxy, o, sgn * n)
                t = time.perf_counter()
                clipped = [c for c in (clip_hull(h, o, sgn * n) for h in hulls) if c is not None]
                t_clip = time.perf_counter() - t
                half = {"side": side, "visual_faces": int(len(hv.faces)), "visual_cut": tv,
                        "proxy_faces": int(len(hp.faces)), "proxy_watertight": bool(hp.is_watertight),
                        "proxy_volume_m3": float(hp.volume) if hp.is_watertight else None, "proxy_cut": tp,
                        "clip_hulls": len(clipped), "clip_s": t_clip, "coacd": {}}
                arrays[f"s{side}_visual_V"], arrays[f"s{side}_visual_F"] = hv.vertices, hv.faces
                arrays[f"s{side}_proxy_V"], arrays[f"s{side}_proxy_F"] = hp.vertices, hp.faces
                arrays.update({f"s{side}_clip_{k}": a for k, a in pack_parts(clipped).items()})
                for nmax in args.coacd:
                    parts, coacd_info = run_coacd(hp, nmax, seed=args.seed)
                    half["coacd"][str(nmax)] = {"hulls": len(parts or []), **coacd_info}
                    arrays.update({f"s{side}_coacd{nmax}_{k}": a for k, a in pack_parts(parts or []).items()})
                rec["halves"].append(half)
            np.savez_compressed(out / f"{name}_cut{k}.npz", **arrays)
            entry["cuts"].append(rec)
            h0, h1 = rec["halves"]
            print(f"  cut {k}: visual {h0['visual_faces']}+{h1['visual_faces']} tris "
                  f"(slice {h0['visual_cut']['slice_s'] + h1['visual_cut']['slice_s']:.3f} s, "
                  f"cap {h0['visual_cut']['cap_s'] + h1['visual_cut']['cap_s']:.3f} s), "
                  f"proxy {h0['proxy_faces']}+{h1['proxy_faces']} (watertight {h0['proxy_watertight']}/"
                  f"{h1['proxy_watertight']}), clip hulls {h0['clip_hulls']}+{h1['clip_hulls']}, coacd "
                  + ", ".join(f"{m}:{h0['coacd'][m]['hulls']}+{h1['coacd'][m]['hulls']} "
                              f"({h0['coacd'][m]['s'] + h1['coacd'][m]['s']:.1f} s)" for m in h0["coacd"]),
                  flush=True)
        manifest["fruits"][name] = entry
    manifest["conditions"]["loadavg_at_end"] = os.getloadavg()
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {out / 'manifest.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="read fruit meshes out of the scene's USD assets (needs pxr)")
    ex.add_argument("--scene", required=True, help="the scene state JSON whose objects to export")
    ex.add_argument("--out", required=True)
    ex.add_argument("--fruits", nargs="+", default=list(FRUITS))
    cu = sub.add_parser("cut", help="cut exported fruits by random planes and build the collision variants")
    cu.add_argument("--out", required=True, help="the export directory; pieces are written next to the exports")
    cu.add_argument("--fruits", nargs="+", default=list(FRUITS))
    cu.add_argument("--n-planes", type=int, default=6)
    cu.add_argument("--proxy-faces", type=int, default=2000)
    cu.add_argument("--coacd", type=int, nargs="+", default=[8])
    cu.add_argument("--min-volume-frac", type=float, default=0.25)
    cu.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    export(args) if args.cmd == "export" else cut(args)


if __name__ == "__main__":
    main()
