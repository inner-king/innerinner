"""썰기 시험용 채소 모양 메쉬(11-12): 당근 토막·애호박(주키니) 토막·감자·오이 토막.

모양은 흔한 크기에 맞춘 단순한 도형이다(사진 복원 아님). 길이 방향 = x(칼질은 x 자리에서 y-z 평면), 단위 m, z-up,
바닥(z=0) 한가운데 원점 — 01_cut_primitive.py --food_mesh 규칙. 끝면은 칼로 자른 토막처럼 평평하고 모서리만 둥글린다.

  python scripts/cutting/food/make_veg_meshes.py [--out data/veg/meshes]

- 당근: 길이 75mm, 지름 29 → 21mm 로 가늘어짐(굵은 쪽 = -x, 손으로 잡는 쪽). 단면은 조금 찌그러진 원.
  누운 당근은 아랫선이 도마에 닿도록 원뿔 반각(약 3°)만큼 기울인다.
- 애호박(주키니): 길이 72mm, 지름 36 → 34mm, 옅은 세로 능선 5개.
- 감자: 64 x 48 x 40mm 타원체에 낮은 주파수 울퉁불퉁함(±3.5%).
- 오이: 지름 30mm, 길이 70mm 원기둥(01_cut_primitive.py 기본 도형과 같은 크기).
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]


def revolve(xs, rad_fn, n_th=96, fillet=0.0015, n_fil=6):
    """x 축 둘레 회전체(끝면 평평 + 모서리 둥글림). rad_fn(x, th) → 반지름(m)."""
    L0, L1 = xs[0], xs[-1]
    th = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    rings = []  # (x, 반지름 줄이는 양)
    for k in range(n_fil, 0, -1):  # 왼쪽 끝 둥근 모서리
        ph = np.pi / 2 * k / n_fil
        rings.append((L0 + fillet - fillet * np.sin(ph), fillet - fillet * np.cos(ph)))
    for x in xs:
        if L0 + fillet <= x <= L1 - fillet:
            rings.append((x, 0.0))
    for k in range(1, n_fil + 1):
        ph = np.pi / 2 * k / n_fil
        rings.append((L1 - fillet + fillet * np.sin(ph), fillet - fillet * np.cos(ph)))
    V = []
    for x, dr in rings:
        r = rad_fn(np.full(n_th, x), th) - dr
        V.append(np.stack([np.full(n_th, x), r * np.cos(th), r * np.sin(th)], 1))
    V = np.concatenate(V)
    nr = len(rings)
    F = []
    for i in range(nr - 1):
        for j in range(n_th):
            a, b = i * n_th + j, i * n_th + (j + 1) % n_th
            c, d = a + n_th, b + n_th
            F += [[a, c, b], [b, c, d]]
    c0, c1 = len(V), len(V) + 1
    V = np.vstack([V, [[rings[0][0], 0, 0], [rings[-1][0], 0, 0]]])
    for j in range(n_th):
        F.append([c0, j, (j + 1) % n_th])
        F.append([c1, (nr - 1) * n_th + (j + 1) % n_th, (nr - 1) * n_th + j])
    m = trimesh.Trimesh(V, np.array(F), process=True)
    trimesh.repair.fix_normals(m)
    return m


def place(m, tilt_y=0.0):
    """y 축으로 tilt_y(rad) 돌린 뒤 바닥 z=0, x·y 가운데 0."""
    if tilt_y:
        m.apply_transform(trimesh.transformations.rotation_matrix(tilt_y, [0, 1, 0]))
    lo, hi = m.bounds
    m.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return m


def smooth_noise(P, rng, n_terms=6, amp=1.0):
    """방향 P(단위 벡터)의 낮은 주파수 매끈한 잡음(대략 -1~1)."""
    out = np.zeros(len(P))
    for _ in range(n_terms):
        k = rng.normal(size=3)
        k = k / np.linalg.norm(k) * rng.uniform(1.0, 2.5)
        out += np.sin(P @ k * np.pi + rng.uniform(0, 2 * np.pi))
    return amp * out / np.sqrt(n_terms)


def carrot():
    L, R1, R2 = 0.075, 0.0145, 0.0105
    xs = np.linspace(-L / 2, L / 2, 76)

    def rad(x, th):
        r = R1 + (R2 - R1) * (x + L / 2) / L
        return r * (1 + 0.035 * np.cos(2 * th + 0.4) + 0.015 * np.cos(3 * th + 1.3 + 8 * x))

    return place(revolve(xs, rad, fillet=0.0015), tilt_y=np.arctan((R1 - R2) / L))


def zucchini():
    L, R1, R2 = 0.072, 0.018, 0.017
    xs = np.linspace(-L / 2, L / 2, 73)

    def rad(x, th):
        r = R1 + (R2 - R1) * (x + L / 2) / L
        return r * (1 + 0.010 * np.cos(5 * th + 0.3) + 0.012 * np.cos(2 * th))

    return place(revolve(xs, rad, fillet=0.002), tilt_y=np.arctan((R1 - R2) / L))


def potato(seed=7):
    rng = np.random.default_rng(seed)
    m = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    P = np.asarray(m.vertices)
    s = 1 + 0.035 * smooth_noise(P, rng)
    m.vertices = P * s[:, None] * np.array([0.032, 0.024, 0.020])
    trimesh.repair.fix_normals(m)
    return place(m)


def cucumber():
    # trimesh.creation.cylinder 는 꼭짓점이 양 끝 고리뿐이라, 겉면 그림(render_surface.py)의 평활(Taubin)에서 작은 원판으로
    # 쪼그라들었다(11-12). 다른 채소처럼 1mm 간격 고리로 만든다. 모서리 둥글림 0.5mm(입자 2.5mm 보다 작아 물리 차이 없음).
    L = 0.07
    return place(revolve(np.linspace(-L / 2, L / 2, 71), lambda x, th: np.full_like(x, 0.015), fillet=0.0005))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "Data/food/meshes"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("carrot", carrot), ("zucchini", zucchini), ("potato", potato), ("cucumber", cucumber)):
        m = fn()
        assert m.is_watertight, name
        m.export(out / f"{name}.obj")
        ext = m.extents * 1000
        print(f"{name}: {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm, 부피 {m.volume * 1e6:.1f} cm^3 → {out / (name + '.obj')}")


if __name__ == "__main__":
    main()
