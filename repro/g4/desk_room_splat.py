# SPDX-License-Identifier: Apache-2.0
"""Keep only the desk top a handheld video observed well, and measure the room around it for plain geometry.

A carved background splat (carve_bg_splat.py) still breaks from the robot's hand camera once it looks away from the
desk (repro/README.md, 30.2): the video was filmed from standing height looking down at the desk, so the walls, the
partition and the floor were seen only from far away or in glimpses, and their Gaussians smear or turn into dark
blobs from anywhere else. This keeps the part of the splat the video did see well and leaves the rest to plain walls
and a floor (add_plain_room.py), placed where the video says the room's surfaces are.

Kept: Gaussians that sit on the observed surface (the carve test's margin, max(3 cm, 5 %)) in at least
--min-frames of the 400 training frames, inside a slab around the support surface (OG z = 0, where the objects
rest) and inside the desk's rectangle. A Gaussian that is behind the observed surface in every frame that sees it
is under the desk top and was never constrained; it goes too. So does the foot of every wall found below (within
3 cm of it), which the plain room replaces, and, along an edge with no wall, whatever hangs more than 3 cm below the
desk top: the video saw neither from the robot's side.

The desk rectangle is found from the well-observed Gaussians in the slab: its orientation is the one whose
projections onto the two axes have the sharpest edges, and an edge is where a row or column of 2 cm cells stops
being at least 40 % occupied. The real robot cart filmed next to the desk sits in the same slab but fills only a
short run of each row, so it falls outside.

The room, in the same frame (u along the desk, v towards the wall behind it): the wall behind the desk and the
side walls are the well-observed Gaussians above the desk next to its edges, the floor is the mode of those below
it. A side the video did not see ends a metre beyond the desk and the video's own camera path. Written next to
the output PLY as <out.ply>.room.json, in OG world coordinates, with the splat's own wall and floor colours.

Usage: python desk_room_splat.py <scene_dir> <in.ply> <out.ply> [--min-frames 20] [--plot top.png]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy import ndimage

SH_C0 = 0.28209479177387814
CELL = 0.02         # m, the occupancy grid of the desk fit
WALL_GAP = 0.03     # m in front of an observed wall that belongs to the wall, not the desk
EDGE_BAND = 0.05    # m inside an open edge (no wall there)
EDGE_Z_MIN = -0.03  # OG z below which a Gaussian in that band hangs under the desk top


def coverage(xyz, ext, K, depth, conf):
    """Per Gaussian: frames that see it, frames where it is on the observed surface, the camera centres."""
    n_frames, h, w = depth.shape
    seen = np.zeros(len(xyz), np.int32)
    surf = np.zeros(len(xyz), np.int32)
    centres = []
    for i in range(n_frames):
        centres.append(-ext[i, :3, :3].T @ ext[i, :3, 3])
        cam = xyz @ ext[i, :3, :3].T + ext[i, :3, 3]
        z = cam[:, 2]
        front = z > 0.05
        u = np.full(len(xyz), -1, np.int64)
        r = np.full(len(xyz), -1, np.int64)
        u[front] = np.floor(K[i, 0, 0] * cam[front, 0] / z[front] + K[i, 0, 2]).astype(np.int64)
        r[front] = np.floor(K[i, 1, 1] * cam[front, 1] / z[front] + K[i, 1, 2]).astype(np.int64)
        idx = np.nonzero(front & (u >= 0) & (u < w) & (r >= 0) & (r < h))[0]
        d = depth[i, r[idx], u[idx]]
        ok = (d > 0) & (conf[i, r[idx], u[idx]] >= 2.0)   # the confidence floor step 4 and the depth loss use
        idx, d = idx[ok], d[ok]
        seen[idx] += 1
        surf[idx[np.abs(z[idx] - d) <= np.maximum(0.03, 0.05 * d)]] += 1
    return seen, surf, np.array(centres)


def rot(yaw):
    """World xy -> (u, v)."""
    return np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])


def occupancy(pts):
    """Cells of CELL holding at least two of the points, the grid origin, and each point's cell."""
    lo = pts.min(0)
    ij = np.floor((pts - lo) / CELL).astype(int)
    occ = np.zeros(ij.max(0) + 1, int)
    np.add.at(occ, (ij[:, 0], ij[:, 1]), 1)
    return occ >= 2, lo, ij


def desk_rectangle(xy):
    """Yaw and [u0, u1] x [v0, v1] of the desk top, from the xy of its well-observed Gaussians."""
    occ, lo, ij = occupancy(xy)
    lab, n = ndimage.label(ndimage.binary_closing(occ, iterations=2))
    big = lab == 1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))))
    xy = xy[big[ij[:, 0], ij[:, 1]]]
    sample = xy[np.random.default_rng(0).permutation(len(xy))[:60000]]

    def sharpness(yaw):
        # Rows and columns of a grid laid along the desk's edges are either full or empty.
        occ = occupancy(sample @ rot(yaw).T)[0]
        return (occ.sum(0) ** 2).sum() + (occ.sum(1) ** 2).sum()

    yaws = np.radians(np.arange(0, 90, 0.25))
    yaw = yaws[int(np.argmax([sharpness(a) for a in yaws]))]
    uv = xy @ rot(yaw).T
    if np.ptp(uv[:, 1]) > np.ptp(uv[:, 0]):  # u along the long side
        yaw += np.pi / 2
    occ, lo, _ = occupancy(xy @ rot(yaw).T)
    rows, cols = occ.sum(0), occ.sum(1)
    rv = np.nonzero(rows >= 0.4 * rows.max())[0]
    cu = np.nonzero(cols >= 0.4 * cols.max())[0]
    return yaw, (lo[0] + cu.min() * CELL, lo[0] + (cu.max() + 1) * CELL), \
        (lo[1] + rv.min() * CELL, lo[1] + (rv.max() + 1) * CELL)


def mode(values, lo, hi, step=0.01):
    hist, edges = np.histogram(values, bins=np.arange(lo, hi + step, step))
    return float(edges[int(np.argmax(hist))] + step / 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--min-frames", type=int, default=20, help="frames a kept Gaussian must be on the surface in")
    ap.add_argument("--slab", type=float, nargs=2, default=[-0.10, 0.05], help="OG z range of the desk top")
    ap.add_argument("--margin", type=float, default=0.03, help="kept beyond the desk's edges (m)")
    ap.add_argument("--min-wall", type=int, default=100,
                    help="well-observed Gaussians above the desk next to an edge that make a wall there")
    ap.add_argument("--plot", default=None, help="top and side views of what is kept and the room")
    args = ap.parse_args()
    scene, src, dst = Path(args.scene_dir), Path(args.src), Path(args.dst)

    # The splat lives in the original DA3 frame and was trained against the scale-corrected void-DA3 depth
    # (carve_bg_splat.py); step 6's sidecar takes it into the OG world, where the objects rest on z = 0.
    orig = np.load(scene / "s2_da" / "da" / "exports" / "npz" / "results.npz")
    void = np.load(scene / "auto_bg" / "da3" / "void_sim3.npz")
    side = json.load(open(f"{src}.pose.json"))
    M = np.array(side["M_src_to_og"])
    ply = PlyData.read(str(src))
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    seen, surf, centres = coverage(xyz, orig["extrinsics"].astype(np.float64), orig["intrinsics"].astype(np.float64),
                                   void["depth"], void["conf"])
    g = xyz @ M[:3, :3].T + M[:3, 3]
    cams = centres @ M[:3, :3].T + M[:3, 3]
    rgb = np.clip(0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), 0, 1)
    good = surf >= args.min_frames
    slab = (g[:, 2] > args.slab[0]) & (g[:, 2] < args.slab[1])

    yaw, (u0, u1), (v0, v1) = desk_rectangle(g[good & slab, :2])
    gu, gv = (g[:, :2] @ rot(yaw).T).T
    above = good & (g[:, 2] > 0.10) & (g[:, 2] < 0.60)
    # The wall behind the desk is the side with the most observed surface standing just past an edge.
    near = [(above & (gv > v1 - 0.15) & (gv < v1 + 0.5)).sum(), (above & (gv < v0 + 0.15) & (gv > v0 - 0.5)).sum()]
    if near[1] > near[0]:
        yaw += np.pi
        gu, gv = -gu, -gv
        u0, u1, v0, v1 = -u1, -u0, -v1, -v0
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    cu, cv = (cams[:, :2] @ rot(yaw).T).T

    walls = {}
    m = above & (gu > u0) & (gu < u1) & (gv > v1 - 0.15) & (gv < v1 + 0.5)
    walls["v_max"] = {"at": mode(gv[m], v1 - 0.15, v1 + 0.5) if m.sum() >= args.min_wall else v1,
                      "observed": int(m.sum()) >= args.min_wall, "gaussians": int(m.sum())}
    band = above & (gv > v0) & (gv < walls["v_max"]["at"] - 0.10)
    for key, edge, out in (("u_min", u0, -1), ("u_max", u1, 1)):
        m = band & (out * (gu - edge) > -0.15) & (out * (gu - edge) < 0.5)
        ok = int(m.sum()) >= args.min_wall
        default = max(edge, cu.max()) + 1.0 if out > 0 else min(edge, cu.min()) - 1.0
        walls[key] = {"at": mode(gu[m], *sorted((edge - out * 0.15, edge + out * 0.5))) if ok else default,
                      "observed": ok, "gaussians": int(m.sum())}
    walls["v_min"] = {"at": min(v0, cv.min()) - 1.0, "observed": False, "gaussians": 0}
    low = good & (g[:, 2] < -0.5)
    floor_ok = int(low.sum()) >= 1000
    floor_z = mode(g[low, 2], -2.0, -0.5) if floor_ok else -0.75

    keep = good & slab & (gu > u0 - args.margin) & (gu < u1 + args.margin) & \
        (gv > v0 - args.margin) & (gv < v1 + args.margin)
    kept_rect = int(keep.sum())
    # The desk ends at an observed wall: what stands within WALL_GAP of it is the wall's own foot (the white wall's
    # lower edge, the dark partition's base), which the plain room replaces. From the robot's cameras it showed as
    # a crumpled strip and a dark fringe in front of the plain wall.
    open_edge = gv < v0 + EDGE_BAND
    for key, val, sign, edge in (("v_max", gv, 1, v1), ("u_min", gu, -1, u0), ("u_max", gu, 1, u1)):
        if walls[key]["observed"]:
            keep &= sign * (val - walls[key]["at"]) < -WALL_GAP
        else:
            open_edge |= sign * (val - edge) > -EDGE_BAND
    kept_walls = int(keep.sum())
    # Along an open edge, what hangs below the desk top is seen only from below its edge, as dark fur.
    keep &= ~(open_edge & (g[:, 2] < EDGE_Z_MIN))
    wall_px = above & (np.abs(gv - walls["v_max"]["at"]) < 0.05)
    floor_px = good & (np.abs(g[:, 2] - floor_z) < 0.05)
    room = {
        "comment": "OG world; u = (cos yaw, sin yaw), v = (-sin yaw, cos yaw) in xy, the wall behind the desk at +v. "
                   "Written by repro/g4/desk_room_splat.py, read by repro/g4/add_plain_room.py.",
        "yaw": float(yaw),
        "desk": {"u": [float(u0), float(u1)], "v": [float(v0), float(v1)], "z": list(args.slab)},
        "walls": walls,
        "floor": {"z": floor_z, "observed": floor_ok, "gaussians": int(low.sum())},
        "video_cameras": {"u": [float(cu.min()), float(cu.max())], "v": [float(cv.min()), float(cv.max())],
                          "z": [float(cams[:, 2].min()), float(cams[:, 2].max())]},
        # The splat's own colours (sRGB, what the linear tonemapper shows) of the surfaces the room replaces.
        "colors": {"wall": np.median(rgb[wall_px], 0).round(3).tolist() if wall_px.any() else [0.75] * 3,
                   "floor": np.median(rgb[floor_px], 0).round(3).tolist() if floor_px.any() else [0.35] * 3},
        "splat": {"src": str(src), "gaussians": len(g), "kept": int(keep.sum()), "min_frames": args.min_frames,
                  "never_on_surface_in_slab": int((slab & (surf == 0)).sum()),
                  "in_desk_rectangle": kept_rect, "dropped_at_walls": kept_rect - kept_walls,
                  "dropped_under_open_edges": kept_walls - int(keep.sum())},
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v[keep], "vertex")], text=False).write(str(dst))
    shutil.copy2(f"{src}.pose.json", f"{dst}.pose.json")
    Path(f"{dst}.room.json").write_text(json.dumps(room, indent=1))
    print(f"desk {u1 - u0:.2f} x {v1 - v0:.2f} m at yaw {np.degrees(yaw):.1f} deg; "
          f"walls " + ", ".join(f"{k} {w['at']:+.2f}{'' if w['observed'] else ' (default)'}" for k, w in walls.items())
          + f"; floor z {floor_z:+.3f}{'' if floor_ok else ' (default)'}")
    print(f"kept {keep.sum()} / {len(g)} Gaussians ({keep.mean():.0%}) -> {dst}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw={"width_ratios": [1.3, 1]})
        sub = np.random.default_rng(0).permutation(len(g))[:250000]
        drop, kept = sub[~keep[sub]], sub[keep[sub]]
        box_u = [walls["u_min"]["at"], walls["u_max"]["at"]]
        box_v = [walls["v_min"]["at"], walls["v_max"]["at"]]
        for ax, (a, b), lab in ((axs[0], (gu, gv), "u (m), along the desk"), (axs[1], (gv, g[:, 2]), "v (m)")):
            ax.scatter(a[drop], b[drop], s=0.3, c=rgb[drop] * 0.35 + 0.65)
            ax.scatter(a[kept], b[kept], s=0.3, c=rgb[kept])
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(lab)
        axs[0].plot([u0, u1, u1, u0, u0], [v0, v0, v1, v1, v0], "b-", lw=1)
        axs[0].plot(box_u + box_u[::-1] + box_u[:1], [box_v[0]] * 2 + [box_v[1]] * 2 + [box_v[0]], "k--", lw=1)
        axs[0].plot(cu, cv, "r-", lw=1)
        axs[0].set_ylabel("v (m), towards the wall behind the desk")
        axs[0].set_title("top: kept (colour), dropped (faded), desk (blue), plain room (dashed), video camera (red)")
        axs[1].plot(box_v + box_v[::-1] + box_v[:1], [floor_z] * 2 + [floor_z + 2.7] * 2 + [floor_z], "k--", lw=1)
        axs[1].plot(cv, cams[:, 2], "r-", lw=1)
        axs[1].set_ylabel("z (m)")
        axs[1].set_title("side")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=60)
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
