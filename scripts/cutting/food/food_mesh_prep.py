"""로봇 에피소드 화면용 식재료 메쉬를 에피소드 전에 미리 만든다 (2026-11-25, 계획 B5, reports/69).

  python scripts/cutting/food/food_mesh_prep.py <src.npz> --out <폴더> [--tex 512] [--no_cuts]

- 들어가는 것: <src.npz> = `x_rest`(입자 처음 자리, 우리 좌표) + `meta`(json 문자열: food_mesh·food_xform·pre_trims·pre_splits·marks·
  category·colors·particle_size). og_veg_cut.py·food_robot_episodes.py 가 쓰는 frames.npz 를 그대로 넣어도 된다.
- 하는 일: 식재료 메쉬(--shape 의 원래 메쉬)를 **계획된 칼질 평면**(meta marks 의 x — 칼이 지나갈 자리)으로 미리 나누고(`pre_splits`),
  조각마다 껍질(꼭짓점 색)·단면(과육·씨 질감)을 만든다. 나누기·색은 다 끝난 실행을 그리는 mesh_export.py·render_surface.py 와 같은 함수다.
  다른 점: 끝난 실행은 칼질 앞뒤 '조각 라벨이 뒤집힌 입자'로 어느 조각을 나눌지 정하지만(split_pieces marks), 에피소드 전에는 결과가 없으므로
  입자를 처음 자리로 평면 앞뒤에 나눈다(split_pieces pre_splits — 채썰기 편과 같은 길). 칼질이 덜 됐으면 이웃 조각 메쉬가 같은 입자들을 따라 붙은 채 그려진다.
- 나오는 것: <out>/mesh.npz(조각별 입자 번호·처음 자리·꼭짓점 이웃 가중치, 부분별 꼭짓점·삼각형·색 또는 uv), mesh.json(부분 목록·질감 이름·걸린 시간),
  tex/*.png(단면 질감). 프레임마다의 변형은 그리는 쪽(food_mesh_view.FoodMesh)이 한다.
- --no_cuts: 칼질 평면으로 나누지 않음(놓기 과제의 통 식재료).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import food_mesh_lib as rs  # noqa: E402  (render_surface.py 에서 메쉬 나누기·질감 부분만 뗀 것)

VERSION = 1  # 메쉬 만드는 방법이 바뀌면 올림(실행기 캐시 열쇠에 들어감)


def plane_splits(meta):
    """계획된 칼질(meta marks) → pre_splits 평면 {n, d}(n·x ≥ d 쪽 = 칼질 앞 조각). 기울인 칼은 marks 의 nrm·z_mm 로."""
    out = []
    for mk in meta.get("marks", []):
        n = np.asarray(mk.get("nrm", [np.cos(np.radians(mk.get("yaw_deg", 0.0))), np.sin(np.radians(mk.get("yaw_deg", 0.0))), 0.0]), float)
        p = np.array([mk["x_mm"], mk.get("y_mm", 0.0), mk.get("z_mm", 0.0) if "nrm" in mk else 0.0]) / 1000.0
        out.append({"n": n.tolist(), "d": float(n @ p)})
    return out


def build(x_rest, meta, tex_px=512, cuts=True):
    """returns (MeshPiece 목록, 쓴 meta)."""
    if not meta.get("food_mesh"):
        raise SystemExit("meta 에 food_mesh 가 없음 — --shape 로 만든 실행만(원기둥 판은 mesh_export.py)")
    m = dict(meta)
    m["pre_splits"] = list(meta.get("pre_splits") or []) + (plane_splits(meta) if cuts else [])
    m["marks"] = []  # 라벨로 나누기 안 함(에피소드 전이라 결과가 없음)
    m.setdefault("z_lift", 0.0)
    m["solver"] = str(meta.get("solver") or "teammate Warp MPM")
    p = float(meta["particle_size"])
    cols = meta.get("colors") or {}
    base_skin = np.array(cols.get("skin", [0.18, 0.37, 0.24]))
    base_flesh = np.array(cols.get("flesh", [0.78, 0.84, 0.72]))
    cat = meta.get("category", meta.get("veg", "cucumber"))
    x_src = x_rest
    if meta.get("color_in_source_frame") and meta.get("food_xform") is not None:
        Ti = np.linalg.inv(np.asarray(meta["food_xform"], float))
        x_src = x_rest @ Ti[:3, :3].T + Ti[:3, 3]
    sec = rs.Section(x_src, p) if cat == "cucumber" else None
    one = np.zeros((1, len(x_rest)), np.int64)
    pieces = rs.build_mesh_pieces(m, x_rest, x_rest[None], one, np.zeros(1, np.int64), sec, base_skin, base_flesh, p, tex_px=tex_px, veg=cat)
    # 칼질 평면 위 입자층(평면에서 입자 간격 절반 안)은 솔버가 어느 쪽 조각에 넣을지 재료마다 다르다(11-25 reports/65 첫 칼질 기록:
    # 오이·당근·감자는 95~99 % 앞 조각, 양파는 0 % — 양파 첫 편 2915 입자 중 753 이 몸통으로 감). 그 층이 조각 메쉬를 반대쪽으로 끌지 않게
    # 어느 조각의 맞춤에도 안 쓴다(입자는 솔버에 그대로, 메쉬만 다음 층부터 따라감).
    planes = [(np.asarray(s["n"], float), float(s["d"])) for s in plane_splits(meta)] if cuts else []
    if planes:
        band = 0.5 * p
        out = []
        for pc in pieces:
            Xr = x_rest[pc.idx]
            far = np.ones(len(Xr), bool)
            for n, d in planes:
                far &= np.abs(Xr @ n - d) > band
            idx = pc.idx[far] if far.sum() >= 8 else pc.idx
            out.append(rs.MeshPiece(idx, pc.parts, x_rest, sigma=2 * p, k=32))
        pieces = out
    return pieces, m


def save(out, pieces, meta_used, info_extra):
    """mesh_export._export 의 '실제 모양' 길과 같은 꼴(껍질 = 꼭짓점 색 skin_vc, 단면 = 질감 cap)."""
    (out / "tex").mkdir(parents=True, exist_ok=True)
    A, info = {}, dict(info_extra, pieces=[])
    for i, pc in enumerate(pieces):
        A[f"p{i}_idx"], A[f"p{i}_Xr"], A[f"p{i}_nb"], A[f"p{i}_w"] = (pc.idx.astype(np.int32), pc.Xr.astype(np.float32),
                                                                      pc.nb.astype(np.int32), pc.w.astype(np.float32))
        parts, off = [], 0
        for nm, V0, Fp, Cp, uvp, img in pc.parts:
            n0 = len(V0)
            if Cp is not None:
                A[f"{nm}_V"], A[f"{nm}_F"], A[f"{nm}_C"] = V0.astype(np.float32), Fp.astype(np.int32), np.clip(Cp, 0, 1).astype(np.float32)
                parts.append(dict(name=nm, kind="skin_vc", v0=off, n=n0, tex=None))
            else:
                A[f"{nm}_V"], A[f"{nm}_F"], A[f"{nm}_uv"] = V0.astype(np.float32), Fp.astype(np.int32), uvp.astype(np.float32)
                Image.fromarray(img).save(out / "tex" / f"{nm}.png")
                parts.append(dict(name=nm, kind="cap", v0=off, n=n0, tex=f"{nm}.png"))
            off += n0
        info["pieces"].append(parts)
    info["pre_splits"] = meta_used["pre_splits"]
    np.savez_compressed(out / "mesh.npz", **A)
    (out / "mesh.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="x_rest·meta 가 든 npz(frames.npz 꼴)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tex", type=int, default=512, help="단면 질감 한 변 화소(학습 화면 224×224 면 512 로 넉넉)")
    ap.add_argument("--no_cuts", action="store_true", help="칼질 평면으로 나누지 않음(통 식재료)")
    a = ap.parse_args()
    t0 = time.time()
    d = np.load(a.src, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    x_rest = np.asarray(d["x_rest"], float)
    pieces, used = build(x_rest, meta, tex_px=a.tex, cuts=not a.no_cuts)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_v = int(sum(len(pt[1]) for pc in pieces for pt in pc.parts))
    n_f = int(sum(len(pt[2]) for pc in pieces for pt in pc.parts))
    info = dict(version=VERSION, src=str(Path(a.src).resolve()), veg=meta.get("veg"), category=meta.get("category"), tex_px=a.tex,
                cuts=not a.no_cuts, n_particles=len(x_rest), vertices=n_v, triangles=n_f, build_s=round(time.time() - t0, 2))
    save(out, pieces, used, info)
    print(json.dumps({k: info[k] for k in ("veg", "vertices", "triangles", "build_s")} | {"pieces": len(pieces)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
