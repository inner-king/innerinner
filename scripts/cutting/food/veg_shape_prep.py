"""채소 실제 모양(Genesis 때 쓴 data/veg/meshes/*.obj)을 팀원 솔버 실행용으로 준비한다 (11-13, reports/37).

  python scripts/cutting/food/veg_shape_prep.py --veg cucumber --keep 0.040 --out data/veg/shapes/cucumber_k40
  python scripts/cutting/food/veg_shape_prep.py --veg potato --slab 0.015 --planks 5 --out data/veg/shapes/potato_slab15

- --keep K: 써는 쪽 끝(+x)에서 K 만큼만 남긴다(계산을 줄임). 잘린 면은 막아 닫힌 메쉬로(manifold3d) — 그림에서는 과육 단면으로 그린다
  (meta pre_planes 로 render_surface 가 단면 질감을 붙임).
- --slab S --planks N(채썰기 시연): 끝에서 S 두께 덩어리를 떼어 z 축으로 90° 돌려(편 면이 y 에 수직) 세우고, y 를 N 장 편으로 나눈 경계를 적는다.
  편썰기한 편들을 원래 순서대로 다시 모아 세운 모양이다.
- --slab S --planks N --stack(11-14, 오이 채썰기 시연): 가운데 S 두께 덩어리를 y 축으로 90° 돌려 눕히고(길이 축 x → z, 편 면이 수평),
  z 를 N 장 편으로 나눈 경계를 적는다. 편썰기한 편 N 장을 원래 순서대로 차곡차곡 쌓은 모양이다(plank_axis = 2).
- 결과: <out>.obj(메쉬), <out>_occ.npz(0.25 mm 칸 점유 격자: 입자를 채울 때 안/밖), <out>.json(경계 상자·잘린 면·편 경계).
좌표: 바닥 z = 0(도마 면). x·y 는 메쉬 좌표 그대로(--slab 이면 돌린 뒤 가운데 0).
"""
import argparse
import json
from pathlib import Path

import manifold3d as mf
import numpy as np
import open3d as o3d
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("--veg", required=True, choices=["cucumber", "zucchini", "carrot", "potato"])
ap.add_argument("--keep", type=float, default=None)
ap.add_argument("--slab", type=float, default=None)
ap.add_argument("--planks", type=int, default=5)
ap.add_argument("--slab_center", type=float, default=None, help="--slab 덩어리 가운데 x [m](주면 그 둘레 S 두께를 뗌 — 양쪽 다 잘린 면). 없으면 끝에서")
ap.add_argument("--trim_end", type=float, default=0.0, help="--keep 과 함께: 써는 쪽 끝(+x)을 이만큼 평평하게 잘라 냄(꼭지 자른 오이, 11-14)")
ap.add_argument("--stack", action="store_true", help="--slab 덩어리를 눕혀 쌓음(편 면 수평, 편 경계 z) — 11-14 채썰기")
ap.add_argument("--voxel", type=float, default=0.25e-3)
ap.add_argument("--out", required=True)
ap.add_argument("--mesh_dir", default=str(Path(__file__).resolve().parents[3] / "Data" / "food" / "meshes"), help="make_veg_meshes.py 결과")
args = ap.parse_args()
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
src = Path(args.mesh_dir) / f"{args.veg}.obj"
m0 = trimesh.load(src, force="mesh")
M = mf.Manifold(mesh=mf.Mesh(vert_properties=np.asarray(m0.vertices, np.float32), tri_verts=np.asarray(m0.faces, np.uint32)))
x1 = float(m0.bounds[1, 0])
info = dict(veg=args.veg, source=str(src.resolve()), pre_planes=[], plank_edges=None, xform=np.eye(4).tolist())
cut = args.slab if args.slab else args.keep
if args.slab and args.slab_center is not None:  # 가운데 덩어리: x ∈ [c − S/2, c + S/2]
    d1, d2 = args.slab_center - args.slab / 2, args.slab_center + args.slab / 2
    A, _ = M.split_by_plane((1.0, 0.0, 0.0), d1)
    _, B = A.split_by_plane((1.0, 0.0, 0.0), d2)
    M = B
    info["pre_planes"] += [dict(n=[1.0, 0.0, 0.0], d=d1, sgn=-1.0), dict(n=[1.0, 0.0, 0.0], d=d2, sgn=1.0)]
elif cut:
    d = x1 - cut
    A, _ = M.split_by_plane((1.0, 0.0, 0.0), d)  # A: x >= d (써는 쪽 끝)
    M = A
    info["pre_planes"].append(dict(n=[1.0, 0.0, 0.0], d=d, sgn=-1.0))  # 잘린 면: 바깥 법선 = −x
    if args.trim_end > 0:  # 11-14: 끝을 평평하게 잘라 낸 면(바깥 법선 +x)
        d2 = x1 - args.trim_end
        _, B = M.split_by_plane((1.0, 0.0, 0.0), d2)
        M = B
        info["pre_planes"].append(dict(n=[1.0, 0.0, 0.0], d=d2, sgn=1.0))
mm = M.to_mesh()
V = np.asarray(mm.vert_properties, float)[:, :3]
F = np.asarray(mm.tri_verts, np.int64)
if args.slab:  # z 축 +90°: (x, y, z) → (−y, x, z). 편 두께 방향(x)이 y 가 된다
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    if args.stack:  # y 축 −90°: (x, y, z) → (−z, y, x). 길이 축(x)이 z, 편 면이 수평
        R = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    V = V @ R.T
    c = 0.5 * (V.min(0) + V.max(0))
    shift = np.array([c[0], c[1], V[:, 2].min()])
    V = V - shift
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -shift
    info["xform"] = T.tolist()  # 원래 메쉬 → 이 좌표(렌더러가 원래 메쉬를 다듬은 뒤 정확한 평면으로 자르도록)
    pp = []
    for pl in info["pre_planes"]:
        n = R @ np.array(pl["n"])
        p0 = R @ (np.array(pl["n"]) * pl["d"]) - shift  # 평면 위 한 점
        pp.append(dict(n=n.tolist(), d=float(n @ p0), sgn=pl["sgn"]))
    info["pre_planes"] = pp
    ax = 2 if args.stack else 1
    y0, y1 = float(V[:, ax].min()), float(V[:, ax].max())
    info["plank_axis"] = ax
    info["stack"] = bool(args.stack)
    info["plank_edges"] = [float(v) for v in np.linspace(y0, y1, args.planks + 1)[1:-1]]
    info["plank_thickness"] = (y1 - y0) / args.planks
mesh = trimesh.Trimesh(V, F, process=True)
assert mesh.is_watertight, "닫힌 메쉬가 아님"
mesh.export(f"{out}.obj")
lo, hi = mesh.bounds
info.update(bounds=[lo.tolist(), hi.tolist()], volume_cm3=float(mesh.volume * 1e6))
# 점유 격자(칸 가운데가 메쉬 안인지)
h = args.voxel
shape = np.ceil((hi - lo) / h).astype(int) + 1
g = [lo[k] + (np.arange(shape[k]) + 0.5) * h for k in range(3)]
P = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
sc = o3d.t.geometry.RaycastingScene()
sc.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)), o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
occ = sc.compute_occupancy(o3d.core.Tensor(P)).numpy().astype(bool).reshape(shape)
np.savez_compressed(f"{out}_occ.npz", lo=lo, h=h, shape=shape, occ=np.packbits(occ.ravel()))
info.update(obj=str(Path(f"{out}.obj").resolve()), occ=str(Path(f"{out}_occ.npz").resolve()), occ_fraction=float(occ.mean()))
Path(f"{out}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
print(json.dumps({k: info[k] for k in ("veg", "bounds", "volume_cm3", "pre_planes", "plank_edges")}, ensure_ascii=False))
