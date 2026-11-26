"""새 식재료 모양(2026-11-20, reports/56): 마늘 쪽·반쪽 양파(겹)·대파 흰 부분(겹)·런천미트·껍질 있는 소시지·돼지 등심 덩어리.

  python scripts/cutting/food/make_food_shapes.py --out data/veg/shapes   (6개 모두)
  ... --only garlic                                                                          (하나만)

무엇: 수식(안이면 음수인 함수)으로 모양을 정의 → 0.25 mm 칸에서 마칭 큐브로 메쉬(.obj) + 점유 격자(_occ.npz) + 정보(.json).
      팀원 솔버 실행기 og_veg_cut.py --shape 가 읽는 형식은 scripts/cutting/food/veg_shape_prep.py 와 같다.
      더한 것: "layers"(처음부터 따로 된 겹 조각 — 양파 비늘·대파 잎집), "shell"(껍질 막 — 소시지, 입자마다 탄성 배율),
      "colors"(겉·속 색).
좌표: 바닥 z = 0(도마 면), y 가운데 0, x = 길이(칼은 +x 끝부터 썬다). 크기 근거: docs/food_material_properties_2026-11-19_new_foods.md 4절.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from skimage import measure

H = 0.25e-3  # 칸 크기 [m]


def garlic(P):
    """마늘 쪽: 길이 26 mm(x), 폭 13 mm(y), 높이 10 mm(z), 평평한 면을 바닥에. 뿌리 쪽(+x)은 잘라 낸 평면(x = +12 mm),
    끝(−x)은 뾰족. 위에서 보면 살짝 휜 초승달(가운데 1.5 mm 휨). 단면은 반타원."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    t = x / 0.013
    g = np.where(t >= 0, np.sqrt(np.clip(1 - (t / 1.25) ** 2, 0, None)), np.sqrt(np.clip(1 - t ** 2, 0, None)))
    w, h = 0.0065 * g + 1e-6, 0.010 * g ** 0.8 + 1e-6
    yc = 0.0015 * (1 - t ** 2)
    f = np.sqrt(((y - yc) / w) ** 2 + (z / h) ** 2) - 1.0
    f = np.maximum(f, -z / 0.003)  # 바닥 평면
    f = np.maximum(f, (x - 0.012) / 0.003)  # 뿌리 쪽 자른 면
    return np.maximum(f, (np.abs(t) - 1.0) * 5)


def onion(P, a=0.030, c=0.028):
    """반쪽 양파(껍질 벗김): 반 가른 면을 바닥에. 반지름 x·y 30 mm, 높이 28 mm. 끝(+x 25 mm)·뿌리(−x 27 mm) 쪽을 평평하게 잘라 냄."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    f = np.sqrt((x / a) ** 2 + (y / a) ** 2 + (z / c) ** 2) - 1.0
    f = np.maximum(f, -z / 0.003)
    f = np.maximum(f, (x - 0.025) / 0.003)
    return np.maximum(f, (-0.027 - x) / 0.003)


def green_onion(P, r=0.010, L=0.060):
    """대파 흰 부분 토막: 지름 20 mm, 길이 60 mm, 도마에 눕힘(축 x, 가운데 높이 10 mm). 양 끝은 자른 면."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    f = np.sqrt(y ** 2 + (z - r) ** 2) / r - 1.0
    return np.maximum(f, (np.abs(x) - L / 2) / 0.003)


def rbox(P, hx, hy, hz, rr):
    """둥근 모서리 상자(바닥 z = 0). hx·hy·hz = 반 길이, rr = 모서리 반지름."""
    q = np.abs(P - np.array([0.0, 0.0, hz])) - np.array([hx, hy, hz]) + rr
    out = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0) - rr
    return out / rr


def sausage(P, r=0.011, L=0.070):
    """껍질 있는 소시지: 지름 22 mm, 길이 70 mm(양 끝 둥근 마개 포함), 도마에 눕힘."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    xc = np.clip(x, -(L / 2 - r), L / 2 - r)
    return np.sqrt((x - xc) ** 2 + y ** 2 + (z - r) ** 2) / r - 1.0


def _smax(a, b, k=0.6):
    """부드러운 최댓값(두 면이 만나는 모서리를 둥글게). a·b 는 mm 단위 거리 비슷한 값."""
    return 0.5 * (a + b + np.sqrt((a - b) ** 2 + k * k))


def green_onion_v2(P, R=0.010):
    """11-20 v2 대파 한 토막(흰 부분 + 초록 잎): x −25~+25 mm, 지름 20 mm, 눕힘(축 높이 10 mm).
    흰 부분(x < +6 mm)은 잎집이 꽉 찬 원기둥. +6 mm 부터 바깥 잎(벽 1.5 mm)과 속 줄기 사이가 비고(고리 모양 빈 곳),
    +10 mm 부터는 속 잎도 관(반지름 4.0~5.3 mm, 벽 1.3 mm)이 되어, 초록 부분은 '관 속에 관'이 든 모양. 양 끝은 자른 면."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    r = np.sqrt(y ** 2 + (z - R) ** 2)
    f = (r - R) / 1e-3
    cav_a = np.maximum(np.maximum(5.3e-3 - r, r - 8.5e-3), 0.006 - x) / 1e-3  # 음수 = 빈 곳 안
    cav_b = np.maximum(r - 4.0e-3, 0.010 - x) / 1e-3
    f = np.maximum(np.maximum(f, -cav_a), -cav_b)
    return np.maximum(f, (np.abs(x) - 0.025) / 1e-3)


def green_onion_v3(P, R=0.010):
    """11-20 저녁 v3(reports/58) 대파 한 토막: green_onion_v2 와 같은 단면 구조인데 초록이 시작하는 자리를 +x 끝 쪽으로 옮겼다.
    +16 mm 부터 바깥 잎(벽 1.5 mm)과 속 줄기 사이가 비고, +19 mm 부터 속 잎도 관(관 속에 관). 그 안쪽(x < +16 mm)은 잎집이 꽉 찬 흰 부분.
    → +x 끝부터 3 mm × 6 번 썰면 초록 관 2편 · 바깥 잎만 관 1편 · 흰 부분 3편(사용자 11-20: 초록 부분과 속 겹을 함께 보이게, 영상 4~6번 칼질)."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    r = np.sqrt(y ** 2 + (z - R) ** 2)
    f = (r - R) / 1e-3
    cav_a = np.maximum(np.maximum(5.3e-3 - r, r - 8.5e-3), 0.016 - x) / 1e-3  # 음수 = 빈 곳 안
    cav_b = np.maximum(r - 4.0e-3, 0.019 - x) / 1e-3
    f = np.maximum(np.maximum(f, -cav_a), -cav_b)
    return np.maximum(f, (np.abs(x) - 0.025) / 1e-3)


def pork_slab(P):
    """11-20 v2 정육점 돼지고기 한 덩이(목살·등심 썬 것): 길이(x) 70, 폭(y) 52, 두께 14~16 mm(윗면 살짝 울퉁불퉁),
    테두리는 둥글고 고르지 않음. +y 쪽 가장자리 약 6 mm 가 지방 띠(색만 — 물리는 같은 재료)."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    ang = np.arctan2(y / 0.026, x / 0.035)
    rad = 1 + 0.05 * np.sin(3 * ang + 0.7) + 0.035 * np.sin(5 * ang + 2.1) + 0.02 * np.sin(7 * ang + 0.3)
    q = (np.abs(x) / (0.035 * rad)) ** 4 + (np.abs(y) / (0.026 * rad)) ** 4
    f_out = (q ** 0.25 - 1.0) * 30.0  # 대략 mm
    top = 0.015 + 0.0012 * np.sin(x / 0.011 + 0.5) * np.cos(y / 0.009) + 0.0008 * np.sin(y / 0.006)
    f = _smax(f_out, (z - top) / 1e-3, 1.5)  # 윗 모서리 둥글게
    return np.maximum(f, -z / 1e-3)



def pork_slab10(P):
    """11-21 reports/59(사용자: "처음 고깃덩이의 두께(높이)를 좀 더 얇게"): pork_slab 과 같은 윤곽·지방 띠, 두께 9~11 mm.
    윗면 울퉁불퉁함은 두께에 맞춰 2/3 로 줄임."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    ang = np.arctan2(y / 0.026, x / 0.035)
    rad = 1 + 0.05 * np.sin(3 * ang + 0.7) + 0.035 * np.sin(5 * ang + 2.1) + 0.02 * np.sin(7 * ang + 0.3)
    q = (np.abs(x) / (0.035 * rad)) ** 4 + (np.abs(y) / (0.026 * rad)) ** 4
    f_out = (q ** 0.25 - 1.0) * 30.0  # 대략 mm
    top = 0.010 + 0.0008 * np.sin(x / 0.011 + 0.5) * np.cos(y / 0.009) + 0.0005 * np.sin(y / 0.006)
    f = _smax(f_out, (z - top) / 1e-3, 1.5)  # 윗 모서리 둥글게
    return np.maximum(f, -z / 1e-3)


FOODS = {
    # 이름: (함수, 경계 상자 lo, hi, 자른 면(렌더가 속살 색으로 그림), 겹, 껍질 막, 색)
    "garlic_clove": (garlic, (-0.0135, -0.009, 0.0), (0.0125, 0.009, 0.0105),
                     [dict(n=[1.0, 0.0, 0.0], d=0.012, sgn=1.0)], None, None,
                     dict(skin=[0.93, 0.89, 0.78], flesh=[0.98, 0.95, 0.84])),
    "onion_half": (onion, (-0.0275, -0.0305, 0.0), (0.0255, 0.0305, 0.0285),
                   [dict(n=[0.0, 0.0, 1.0], d=0.0, sgn=-1.0), dict(n=[1.0, 0.0, 0.0], d=0.025, sgn=1.0),
                    dict(n=[1.0, 0.0, 0.0], d=-0.027, sgn=-1.0)],
                   dict(kind="ellipsoid", center=[0.0, 0.0, 0.0], radii=[0.030, 0.030, 0.028], edges=[0.64, 0.76, 0.88]),
                   None, dict(skin=[0.93, 0.88, 0.70], flesh=[0.96, 0.94, 0.86])),
    "green_onion_white": (green_onion, (-0.0305, -0.0105, 0.0), (0.0305, 0.0105, 0.0205),
                          [dict(n=[1.0, 0.0, 0.0], d=0.030, sgn=1.0), dict(n=[1.0, 0.0, 0.0], d=-0.030, sgn=-1.0)],
                          dict(kind="cylinder", axis=0, center=[0.0, 0.010], edges=[0.0025, 0.005, 0.0075]),
                          None, dict(skin=[0.90, 0.94, 0.82], flesh=[0.97, 0.98, 0.93])),
    "luncheon_block": (lambda P: rbox(P, 0.025, 0.020, 0.015, 0.003), (-0.0255, -0.0205, 0.0), (0.0255, 0.0205, 0.0305),
                       [], None, None, dict(skin=[0.86, 0.56, 0.52], flesh=[0.90, 0.62, 0.58])),
    "sausage_cased": (sausage, (-0.0355, -0.0115, 0.0), (0.0355, 0.0115, 0.0225),
                      [], None, dict(kind="capsule", axis=0, center=[0.0, 0.011], radius=0.011, half_len=0.024, thickness=0.0009),
                      dict(skin=[0.62, 0.30, 0.20], flesh=[0.88, 0.60, 0.55])),
    "pork_loin_block": (lambda P: rbox(P, 0.025, 0.020, 0.015, 0.002), (-0.0255, -0.0205, 0.0), (0.0255, 0.0205, 0.0305),
                        [], None, None, dict(skin=[0.86, 0.50, 0.50], flesh=[0.88, 0.52, 0.52])),
    # 11-20 v2(reports/57): 사용자 "대파 초록 부분(얇은 껍질만 있는 관)·속 겹", "생고기는 실제 파는 고기처럼 얇고 질기게"
    "green_onion_v2": (green_onion_v2, (-0.0255, -0.0105, 0.0), (0.0255, 0.0105, 0.0205),
                       [dict(n=[1.0, 0.0, 0.0], d=0.025, sgn=1.0), dict(n=[1.0, 0.0, 0.0], d=-0.025, sgn=-1.0)],
                       None, None, dict(skin=[0.93, 0.95, 0.88], flesh=[0.96, 0.97, 0.92])),
    # 11-20 저녁 v3(reports/58): 초록 시작을 +16/+19 mm 로(3 mm × 6 칼질에 초록·흰 부분이 다 들어오게)
    "green_onion_v3": (green_onion_v3, (-0.0255, -0.0105, 0.0), (0.0255, 0.0105, 0.0205),
                       [dict(n=[1.0, 0.0, 0.0], d=0.025, sgn=1.0), dict(n=[1.0, 0.0, 0.0], d=-0.025, sgn=-1.0)],
                       None, None, dict(skin=[0.93, 0.95, 0.88], flesh=[0.96, 0.97, 0.92])),
    "pork_slab": (pork_slab, (-0.0395, -0.0300, 0.0), (0.0395, 0.0300, 0.0175),
                  [], None, None, dict(skin=[0.82, 0.44, 0.44], flesh=[0.86, 0.48, 0.47])),
    # 11-21 reports/59: 더 얇은 덩이(두께 10 mm)
    "pork_slab10": (pork_slab10, (-0.0395, -0.0300, 0.0), (0.0395, 0.0300, 0.0125),
                    [], None, None, dict(skin=[0.82, 0.44, 0.44], flesh=[0.86, 0.48, 0.47])),
}


def build(name, out_dir):
    fn, lo, hi, planes, layers, shell, colors = FOODS[name]
    lo, hi = np.array(lo) - 2 * H, np.array(hi) + 2 * H
    shape = np.ceil((hi - lo) / H).astype(int) + 1
    g = [lo[k] + (np.arange(shape[k]) + 0.37) * H for k in range(3)]  # 격자점이 평면(z = 0 등) 위에 딱 놓이면 마칭 큐브가 겹친 면을 만든다
    P = np.stack(np.meshgrid(*g, indexing="ij"), -1)
    f = fn(P)
    f = np.where(np.abs(f) < 1e-3, 1e-3, f)  # 11-20: 1e-6 은 mm 단위 함수에서 너무 작아 꼭짓점이 겹침  # 격자점 값이 0 이면 꼭짓점이 겹쳐 닫히지 않는 모서리가 생김
    f[0, :, :] = f[-1, :, :] = f[:, 0, :] = f[:, -1, :] = f[:, :, 0] = f[:, :, -1] = 1.0  # 닫힌 메쉬
    v, faces, _, _ = measure.marching_cubes(f, 0.0, spacing=(H, H, H))
    v = v + lo + 0.37 * H  # 바닥 평면은 f = −z/3mm 라 0 을 지나는 곳이 정확히 z = 0
    mesh = trimesh.Trimesh(v, faces[:, ::-1], process=True)
    if mesh.volume < 0:
        mesh.invert()
    assert mesh.is_watertight, f"{name}: 닫힌 메쉬가 아님"
    base = Path(out_dir) / name
    mesh.export(f"{base}.obj")
    mlo, mhi = mesh.bounds
    # 점유 격자(칸 가운데가 메쉬 안인지) — veg_shape_prep.py 와 같은 방식
    oshape = np.ceil((mhi - mlo) / H).astype(int) + 1
    og = [mlo[k] + (np.arange(oshape[k]) + 0.5) * H for k in range(3)]
    Q = np.stack(np.meshgrid(*og, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)), o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    occ = sc.compute_occupancy(o3d.core.Tensor(Q)).numpy().astype(bool).reshape(oshape)
    np.savez_compressed(f"{base}_occ.npz", lo=mlo, h=H, shape=oshape, occ=np.packbits(occ.ravel()))
    info = dict(veg=name, source=str(Path(f"{base}.obj").resolve()), pre_planes=planes, plank_edges=None, xform=np.eye(4).tolist(),
                bounds=[mlo.tolist(), mhi.tolist()], volume_cm3=float(mesh.volume * 1e6), obj=str(Path(f"{base}.obj").resolve()),
                occ=str(Path(f"{base}_occ.npz").resolve()), occ_fraction=float(occ.mean()), colors=colors)
    if layers:
        info["layers"] = layers
    if shell:
        info["shell"] = shell
    Path(f"{base}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    ext = (mhi - mlo) * 1000
    print(f"{name}: {ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} mm, 부피 {info['volume_cm3']:.1f} cm³"
          + (f", 겹 {len(layers['edges']) + 1}" if layers else "") + (", 껍질 막" if shell else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Data/food/shapes")
    ap.add_argument("--only", default=None, choices=sorted(FOODS))
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for nm in ([a.only] if a.only else FOODS):
        build(nm, a.out)
