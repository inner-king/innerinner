"""Food mesh pieces for the robot episode view (scripts/cutting/food): pre-split the food mesh by the planned cut planes,
skin vertex colours and cut-face textures (flesh, seeds), and per-piece fitting to the particles. Extracted from the author's
render_surface.py without its Open3D renderer, surface reconstruction and video code. Needs numpy, scipy, trimesh, PIL and
manifold3d (pip install manifold3d); Open3D is optional (exact depth below the skin; a voxel distance transform otherwise).
Comments are in Korean (original project)."""
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter1d


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def mix(c0, c1, w):
    c0, c1 = np.asarray(c0, float), np.asarray(c1, float)
    c0, c1 = (c0[None] if c0.ndim == 1 else c0), (c1[None] if c1.ndim == 1 else c1)
    return c0 * (1 - w[:, None]) + c1 * w[:, None]


def hash01(*ints):
    """정수 격자 좌표 → 0~1 의사 난수(무늬용)."""
    h = np.zeros_like(ints[0], dtype=np.uint64)
    for k, v in zip((73856093, 19349663, 83492791), ints):
        h ^= (v.astype(np.int64).astype(np.uint64) * np.uint64(k))
    h = (h ^ (h >> np.uint64(13))) * np.uint64(0x5bd1e995)
    return ((h ^ (h >> np.uint64(15))) & np.uint64(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)


def vnoise(p, scale):
    """부드러운 3D 값 잡음(-1~1). p (M,3) m, scale m."""
    g = p / scale
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    f = f * f * (3 - 2 * f)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w * hash01(i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz)
    return out * 2 - 1


class Section:
    """처음 모양에서 x 축(긴 축)을 따라 단면 중심·반지름 표. 원기둥꼴 식재료의 단면 좌표(rho, theta)용."""

    def __init__(self, X, p, step=0.002):
        lo, hi = X[:, 0].min(), X[:, 0].max()
        self.xs = np.arange(lo, hi + step, step)
        tab = []
        for x in self.xs:
            m = np.abs(X[:, 0] - x) < max(step, p)
            P = X[m] if m.sum() >= 8 else X[np.argsort(np.abs(X[:, 0] - x))[:8]]
            a, b = P[:, 1:].min(0) - p / 2, P[:, 1:].max(0) + p / 2
            tab.append([*(a + b) / 2, *np.maximum((b - a) / 2, p)])
        self.tab = gaussian_filter1d(np.array(tab), 1.5, axis=0)

    def coords(self, R):
        c = np.stack([np.interp(R[:, 0], self.xs, self.tab[:, k]) for k in range(4)], 1)
        u, v = (R[:, 1] - c[:, 0]) / c[:, 2], (R[:, 2] - c[:, 1]) / c[:, 3]
        return np.hypot(u, v), np.arctan2(v, u)

    def radius(self, x):
        """x 자리 단면의 평균 반지름(m)."""
        return 0.5 * (np.interp(x, self.xs, self.tab[:, 2]) + np.interp(x, self.xs, self.tab[:, 3]))


def srgb(c):
    return np.clip(c, 0, 1)


def cucumber_warts(R, th):
    """껍질의 작은 혹 자리: (혹 중심까지 거리 / 혹 반지름, 혹이 있는 칸인지). 무늬와 돌기가 같은 자리를 쓴다."""
    cell_x, cell_t = np.floor(R[:, 0] / 0.006), np.floor((th + np.pi) / (2 * np.pi / 24))
    ix, it = cell_x.astype(np.int64), cell_t.astype(np.int64)
    k5 = np.full(len(th), 5, np.int64)
    cx = (cell_x + 0.25 + 0.5 * hash01(ix, it, k5 + 1)) * 0.006
    ct = (cell_t + 0.25 + 0.5 * hash01(ix, it, k5 + 2)) * (2 * np.pi / 24) - np.pi
    dd = np.hypot((R[:, 0] - cx) / 0.0006, np.angle(np.exp(1j * (th - ct))) * 0.021 / 0.0006)
    return dd, hash01(ix, it, k5) < 0.22


def cucumber_colors(R, depth, skin, sec, base_skin, base_flesh):
    """R: 처음 위치(m), depth: 처음 겉면에서 깊이(m), skin: 껍질 입자 비율(0~1) → sRGB(0~1).
    색·크기는 Chop&Learn 사진의 오이에 맞췄다(사진 색에는 초록 도마 반사가 섞여 있어 base_skin·base_flesh 는 안 쓴다).
    단면: 크림색 과육, 0.6mm 짙은 껍질 테두리와 그 안쪽 2mm 초록 층, 과육보다 조금 어둡고 반투명한 세 갈래 씨 자리,
    가운데 옅은 세 칸막이와 그 양쪽에서 바깥을 향한 흰 씨, 씨 자리 둘레의 옅은 관다발 점.
    껍질: 짙은 초록 + 옅은 세로 줄·얼룩 + 드문드문 연한 혹."""
    rho, th = sec.coords(R)
    x = R[:, 0]
    rad = sec.radius(x)
    nn = len(x)
    n1 = vnoise(R, 0.004)
    n2 = vnoise(R + 0.37, 0.0012)
    # 과육: 사진 단면(가운데~바깥 평균 sRGB 0.75~0.78, 0.81~0.84, 0.69~0.74)보다 조명만큼 밝게
    col = mix([0.88, 0.90, 0.79], [0.84, 0.88, 0.73], smoothstep(0.55, 0.92, rho))
    col = col * (1 + 0.018 * n1 + 0.012 * n2)[:, None]
    # 씨 자리: 세 갈래로 둥글게 부푼 반투명한 젤리. 갈래 방향은 길이를 따라 조금 돈다
    a0 = np.pi / 2 + 0.25 * np.sin(x / 0.035)
    edge = 0.50 + 0.045 * np.cos(3 * (th - a0)) + 0.03 * n1
    col = mix(col, [0.80, 0.85, 0.72], smoothstep(edge + 0.07, edge - 0.07, rho) * 0.6)
    col = mix(col, [0.90, 0.92, 0.84], smoothstep(0.12, 0.0, rho) * 0.35)  # 가운데 심
    # 관다발: 씨 자리 바로 바깥 고리의 옅은 점(길이 방향으로 뻗은 관이라 단면에서는 점)
    nb = 14
    kb = np.round((th - a0) / (2 * np.pi / nb))
    jb = hash01((kb % nb).astype(np.int64), np.full(nn, 3, np.int64)) - 0.5
    ab = a0 + (kb + 0.3 * jb) * 2 * np.pi / nb
    db = np.hypot(np.angle(np.exp(1j * (th - ab))) * rho, rho - (edge + 0.08 + 0.04 * jb)) * rad
    col = mix(col, [0.74, 0.83, 0.63], smoothstep(0.0007, 0.0002, db) * 0.45)
    u, v = rho * np.cos(th), rho * np.sin(th)
    for k in range(3):
        a = a0 + k * 2 * np.pi / 3
        # 칸막이: 가운데에서 씨 자리 끝 가까이까지 가늘고 옅게
        d = np.abs(np.angle(np.exp(1j * (th - a)))) * rho * rad
        arm = smoothstep(0.0009, 0.0003, d + 0.0003 * n2) * smoothstep(edge - 0.03, edge - 0.15, rho)
        col = mix(col, [0.90, 0.92, 0.83], arm * 0.25)
        # 씨: 칸막이 양쪽에 세 줄씩, 길이 방향 3mm 간격(줄마다 엇갈림). 단면에 10~15개가 몰려 보인다(사진)
        for j, (side, r0) in enumerate((s, r) for s in (-1, 1) for r in (0.20, 0.31, 0.42)):
            pitch, off = 0.0030, 0.25 * j
            xi = np.floor(x / pitch + off).astype(np.int64)
            hk = lambda s: hash01(xi, np.full(nn, 8 * k + j, np.int64), np.full(nn, s, np.int64))
            xc = (xi + 0.5 - off) * pitch + (hk(1) - 0.5) * 0.0010
            ac = a + side * (0.105 / r0 + 0.10 * (hk(2) - 0.5))  # 칸막이에서 2mm 남짓 떨어진 자리
            rc = r0 + 0.05 * (hk(3) - 0.5)
            e_dir = ac + side * 0.3  # 씨 끝이 칸막이에서 조금 벌어진다
            ou, ov = (u - rc * np.cos(ac)) * rad, (v - rc * np.sin(ac)) * rad
            along = ou * np.cos(e_dir) + ov * np.sin(e_dir)
            across = -ou * np.sin(e_dir) + ov * np.cos(e_dir)
            e = (along / 0.0020) ** 2 + (across / 0.0010) ** 2 + ((x - xc) / 0.0016) ** 2
            e = np.where(hk(4) < 0.85, e, 9.0)  # 빈 자리도 있다
            col = mix(col, [0.74, 0.80, 0.65], smoothstep(2.2, 1.0, e) * 0.4)  # 씨를 감싼 젤리
            col = mix(col, [0.93, 0.93, 0.84], smoothstep(1.0, 0.45, e) * 0.9)
    # 껍질 바로 안쪽 초록 층(2mm 안에서 과육으로 옅어짐)과 짙은 껍질 테두리(0.6mm)
    col = mix(col, [0.52, 0.68, 0.36], smoothstep(0.0026, 0.0005, depth + 0.0004 * n1) * 0.9)
    w_rim = smoothstep(0.0008, 0.0003, depth + 0.0002 * n2) * 0.85
    # 껍질: 짙은 초록 + 옅고 곧은 세로 줄과 얼룩 + 드문드문 연한 혹. 사진 색(도마 반사 뺀다며 0.10·0.19·0.09)은
    # 렌더에서 탁한 올리브·검은 고무처럼 보여(사용자 확인) 밝혔고, 물결 줄무늬는 밝히면 수박처럼 도드라져 약하게
    stripe = smoothstep(0.6, 0.95, np.cos(10 * th + 0.6 * vnoise(R + 0.9, 0.012)))
    blot = smoothstep(0.2, 0.9, vnoise(R + 1.7, 0.007))
    sk = mix([0.13, 0.26, 0.10], [0.20, 0.34, 0.15], stripe * 0.15 + blot * 0.10)
    sk = sk * (1 + 0.08 * n2)[:, None]
    dd, has = cucumber_warts(R, th)
    sk = mix(sk, [0.42, 0.48, 0.30], smoothstep(1.0, 0.3, dd) * has * 0.45)
    w_skin = np.maximum(smoothstep(0.30, 0.65, skin), w_rim)
    return srgb(mix(col, sk, w_skin))


def generic_colors(R, depth, skin, base_skin, base_flesh):
    n1 = vnoise(R, 0.004)
    flesh = np.asarray(base_flesh, float)[None] * (1 + 0.03 * n1)[:, None]
    sk = np.asarray(base_skin, float)[None] * (1 + 0.08 * vnoise(R + 0.3, 0.0015))[:, None]
    w = np.maximum(smoothstep(0.30, 0.65, skin), smoothstep(0.0024, 0.0015, depth))
    return srgb(mix(flesh, sk, w))


def tofu_colors(R, depth, skin, base_skin, base_flesh):
    """두부(10-23): 속은 흰빛에 아주 작은 구멍(0.7mm 잡음의 어두운 점), 겉면 0.5~1mm 는 조금 누런빛. generic_colors 와 같은 꼴."""
    pore = smoothstep(0.35, 0.75, vnoise(R, 0.0007))
    flesh = np.asarray(base_flesh, float)[None] * (1 + 0.02 * vnoise(R, 0.003) - 0.10 * pore)[:, None]
    sk = np.asarray(base_skin, float)[None] * (1 + 0.03 * vnoise(R + 0.3, 0.002) - 0.04 * pore)[:, None]
    w = np.maximum(smoothstep(0.30, 0.65, skin), smoothstep(0.0010, 0.0005, depth))
    return srgb(mix(flesh, sk, w))


def carrot_colors(R, depth, skin, sec):
    """당근(11-12): 단면 = 짙은 주황 겉살(사부) + 연한 노란 주황 심(목부, 반지름 55% 안, 가는 방사선) + 그 경계의 옅은 고리(형성층).
    껍질 = 조금 탁한 주황 + 0.3mm 짙은 테두리 + 가로 주름(2~4mm 간격 옅은 줄, 피목). 색은 흔한 당근 사진 어림(측정 아님)."""
    rho, th = sec.coords(R)
    n1, n2 = vnoise(R, 0.003), vnoise(R + 0.41, 0.0009)
    edge = 0.55 + 0.012 * np.cos(5 * th + 6 * R[:, 0]) + 0.012 * np.cos(3 * th + 1.1) + 0.025 * n1  # 심 경계(조금 물결)
    col = mix([0.93, 0.44, 0.07], [0.96, 0.52, 0.11], smoothstep(0.95, 0.70, rho))
    core = mix([0.98, 0.70, 0.30], [0.97, 0.62, 0.21], smoothstep(0.05, 0.45, rho))
    ray = smoothstep(0.75, 1.0, np.cos(28 * th + 3 * n1)) * smoothstep(0.08, 0.3, rho)  # 심의 방사선
    core = core * (1 + 0.05 * ray)[:, None]
    col = mix(col, core, smoothstep(edge + 0.02, edge - 0.02, rho))
    ring = np.exp(-((rho - edge) / 0.018) ** 2)
    col = mix(col, [0.99, 0.76, 0.38], ring * 0.55)
    col = col * (1 + 0.025 * n1 + 0.02 * n2)[:, None]
    # 껍질
    groove = smoothstep(0.88, 1.0, np.cos(R[:, 0] / 0.0031 * 2 * np.pi + 2.2 * vnoise(R, 0.010)))
    groove = groove * smoothstep(0.1, 0.6, vnoise(R + 4.2, 0.005))  # 둘레를 다 돌지 않는 짧은 주름
    sk = mix([0.86, 0.40, 0.09], [0.78, 0.34, 0.08], 0.5 + 0.5 * vnoise(R + 0.7, 0.004))
    sk = mix(sk, [0.70, 0.30, 0.07], groove * 0.30)
    sk = sk * (1 + 0.06 * n2)[:, None]
    w_rim = smoothstep(0.0006, 0.0002, depth + 0.0001 * n2) * 0.8
    w = np.maximum(smoothstep(0.30, 0.65, skin), w_rim)
    w = w * (1 - smoothstep(0.90, 0.82, rho))  # 토막 끝면(처음부터 잘린 면, 반지름 안쪽)은 껍질 아닌 단면 색
    return srgb(mix(col, sk, w))


def zucchini_colors(R, depth, skin, sec):
    """애호박(주키니, 11-12): 단면 = 옅은 크림색 과육 + 껍질 안쪽 1.5mm 연두 층 + 가운데 조금 반투명한 씨 자리(세 갈래, 작은 흰 씨).
    껍질 = 짙은 초록 + 옅은 세로 줄 + 연두 반점. 색은 흔한 주키니 사진 어림(측정 아님)."""
    rho, th = sec.coords(R)
    x = R[:, 0]
    rad = sec.radius(x)
    n1, n2 = vnoise(R, 0.004), vnoise(R + 0.37, 0.0011)
    col = mix([0.95, 0.95, 0.84], [0.92, 0.94, 0.78], smoothstep(0.6, 0.95, rho))
    col = col * (1 + 0.015 * n1 + 0.012 * n2)[:, None]
    a0 = 0.4 + 0.2 * np.sin(x / 0.04)
    edge = 0.42 + 0.04 * np.cos(3 * (th - a0)) + 0.025 * n1
    col = mix(col, [0.86, 0.89, 0.74], smoothstep(edge + 0.06, edge - 0.06, rho) * 0.55)  # 씨 자리(조금 어둡고 반투명)
    u, v = rho * np.cos(th), rho * np.sin(th)
    nn = len(x)
    for k in range(3):
        a = a0 + k * 2 * np.pi / 3
        for j, r0 in enumerate((0.18, 0.30)):
            pitch = 0.0035
            xi = np.floor(x / pitch + 0.37 * j).astype(np.int64)
            hk = lambda s: hash01(xi, np.full(nn, 5 * k + j, np.int64), np.full(nn, s, np.int64))
            xc = (xi + 0.5 - 0.37 * j) * pitch + (hk(1) - 0.5) * 0.001
            for side in (-1, 1):
                ac = a + side * (0.09 / r0)
                ou, ov = (u - r0 * np.cos(ac)) * rad, (v - r0 * np.sin(ac)) * rad
                along = ou * np.cos(ac) + ov * np.sin(ac)
                across = -ou * np.sin(ac) + ov * np.cos(ac)
                e = (along / 0.0017) ** 2 + (across / 0.0008) ** 2 + ((x - xc) / 0.0013) ** 2
                e = np.where(hk(2 + side) < 0.8, e, 9.0)
                col = mix(col, [0.97, 0.96, 0.88], smoothstep(1.0, 0.4, e) * 0.75)
    # 껍질 바로 안쪽 연두 층과 짙은 테두리. 겉면 점(skin=1)은 처음 겉면이라 깊이가 0 이므로, 토막 끝면에서는 반지름 쪽 깊이로 잰다
    dep = np.where(skin > 0.5, np.maximum((1 - rho) * rad, 0.0), depth)
    col = mix(col, [0.60, 0.76, 0.40], smoothstep(0.0022, 0.0004, dep + 0.0003 * n1) * 0.85)
    w_rim = smoothstep(0.0007, 0.0003, dep + 0.0002 * n2) * 0.9
    stripe = smoothstep(0.55, 0.95, np.cos(14 * th + 0.8 * vnoise(R + 0.9, 0.015)))
    speck = smoothstep(0.55, 0.85, vnoise(R + 2.3, 0.0011))
    sk = mix([0.09, 0.22, 0.08], [0.20, 0.36, 0.14], stripe * 0.35)
    sk = mix(sk, [0.45, 0.58, 0.30], speck * 0.55)
    sk = sk * (1 + 0.07 * n2)[:, None]
    w = np.maximum(smoothstep(0.30, 0.65, skin), w_rim)
    w = w * (1 - smoothstep(0.90, 0.82, rho))  # 토막 끝면(처음부터 잘린 면, 반지름 안쪽)은 껍질 아닌 단면 색
    return srgb(mix(col, sk, w))


def potato_colors(R, depth, skin):
    """감자(11-12): 단면 = 옅은 노란 크림색 + 겉면에서 3~4mm 안쪽의 옅은 관다발 고리. 껍질 = 연갈색 + 짙은 얼룩·작은 눈(싹 자리)."""
    n1, n2 = vnoise(R, 0.004), vnoise(R + 0.53, 0.0010)
    col = np.asarray([0.97, 0.91, 0.66])[None] * (1 + 0.02 * n1 + 0.015 * n2)[:, None]
    ring = np.exp(-((depth - 0.0036 - 0.0005 * n1) / 0.00045) ** 2)
    col = mix(col, [0.90, 0.85, 0.62], ring * 0.6)
    col = mix(col, [0.93, 0.86, 0.60], smoothstep(0.0012, 0.0004, depth) * 0.6)
    sk = mix([0.70, 0.54, 0.34], [0.58, 0.43, 0.26], smoothstep(0.1, 0.8, vnoise(R + 1.1, 0.005)))
    sk = mix(sk, [0.45, 0.33, 0.20], smoothstep(0.6, 0.9, vnoise(R + 3.1, 0.0012)) * 0.6)  # 짙은 점
    eye = smoothstep(0.82, 0.95, vnoise(R + 5.7, 0.006))
    sk = mix(sk, [0.40, 0.28, 0.17], eye * 0.7)
    sk = sk * (1 + 0.06 * n2)[:, None]
    w_rim = smoothstep(0.0005, 0.0002, depth) * 0.85
    w = np.maximum(smoothstep(0.30, 0.65, skin), w_rim)
    return srgb(mix(col, sk, w))


def _new_food_colors(cat, R, depth, skin, base_skin, base_flesh):
    """11-20 새 식재료(reports/56, scripts/cutting/food/make_food_shapes.py 모양) 색. R = 처음 자리(우리 좌표, m)."""
    n1 = vnoise(R, 0.004)
    flesh = np.asarray(base_flesh, float)[None] * (1 + 0.02 * n1)[:, None]
    sk = np.asarray(base_skin, float)[None] * (1 + 0.05 * vnoise(R + 0.3, 0.0015))[:, None]
    w_skin = np.maximum(smoothstep(0.30, 0.65, skin), smoothstep(0.0012, 0.0006, depth))
    if cat == "onion_half":  # 11-20 저녁 v3(reports/58): 비늘 한 장 약 3 mm — 막 줄(가늘고 짙음) + 비늘 안쪽으로 갈수록 비쳐 보이는 색
        rho = np.sqrt((R[:, 0] / 0.030) ** 2 + (R[:, 1] / 0.030) ** 2 + (R[:, 2] / 0.028) ** 2)
        sp = 0.10 + 0.004 * vnoise(R + 2.4, 0.010)  # 비늘 두께 약 3 mm(반지름 30 mm 의 0.10), 조금씩 다름
        ph = np.mod(rho / sp, 1.0)  # 0 = 비늘 안쪽 면(막), 1 = 바깥쪽 면
        flesh = np.asarray([0.94, 0.92, 0.82])[None] * (1 + 0.015 * n1)[:, None]
        flesh = mix(flesh, [0.86, 0.85, 0.70], smoothstep(0.55, 0.0, ph) * 0.35)  # 비늘 안쪽 반: 물기 많은 속살이 비쳐 조금 짙고 노르스름
        flesh = mix(flesh, [0.75, 0.74, 0.55], smoothstep(0.07, 0.0, np.minimum(ph, 1 - ph)) * 0.85)  # 비늘 사이 막
        flesh = mix(flesh, [0.97, 0.96, 0.90], smoothstep(0.25, 0.0, rho) * 0.5)  # 가운데 어린 비늘은 하얗게
        flesh = mix(flesh, [0.90, 0.86, 0.62], smoothstep(0.80, 0.98, rho) * 0.45)  # 바깥 비늘은 노르스름
        veins = vnoise(np.stack([np.arctan2(R[:, 2], R[:, 1]) * 0.03, R[:, 0] * 0.15, rho * 0.03], 1), 0.0009)  # 겉 잎맥(뿌리–끝 = x 축 방향)
        sk = np.asarray([0.90, 0.85, 0.64])[None] * (1 + 0.05 * veins + 0.02 * n1)[:, None]
    elif cat == "green_onion_white":  # 잎집 겹 테두리(1.25 mm 마다), 바깥쪽일수록 옅은 연두
        r = np.hypot(R[:, 1], R[:, 2] - 0.010)
        ph = np.mod(r / 0.00125, 1.0)
        flesh = mix(flesh, [0.82, 0.88, 0.70], smoothstep(0.12, 0.0, np.minimum(ph, 1 - ph)) * 0.45)
        flesh = mix(flesh, [0.80, 0.90, 0.66], smoothstep(0.004, 0.010, r) * 0.35)
    elif cat == "pork_loin_block":  # 근섬유 결(x 방향으로 긴 잡음) + 가는 흰 지방 줄
        fib = vnoise(R * np.array([0.15, 1.0, 1.0]), 0.0006)
        flesh = flesh * (1 + 0.06 * fib)[:, None]
        flesh = mix(flesh, [0.97, 0.90, 0.86], smoothstep(0.80, 0.90, vnoise(R * np.array([0.3, 1.0, 1.0]) + 2.1, 0.0025)) * 0.8)
        w_skin = w_skin * 0.0  # 덩어리 겉도 잘린 면이라 같은 살색
    elif cat in ("green_onion_v2", "green_onion_v3"):
        # v2(11-20): 흰 부분 잎집 고리 → +3~+11 mm 에서 초록 관. v3(11-20 저녁, reports/58): 모양의 초록 시작을 +16/+19 mm 로 옮긴 토막
        # (3 mm × 6 칼질 중 셋이 흰 부분) + 색을 실물 쪽으로(짙은 청록 겉·흰 가루 얼룩, 잎 벽 안팎 색, 흰 부분 겹 줄 진하게).
        x, r = R[:, 0], np.hypot(R[:, 1], R[:, 2] - 0.010)
        x_w, x_g = (0.003, 0.011) if cat == "green_onion_v2" else (0.010, 0.019)  # 흰 → 초록 색이 바뀌는 x 범위(겉)
        g = smoothstep(x_w, x_g, x)
        ring = r / 0.0011
        band = np.floor(ring) % 2
        white = np.asarray([0.94, 0.95, 0.88])[None] * (1 + 0.06 * (band - 0.5) + 0.02 * n1)[:, None]
        ph = np.mod(ring, 1.0)
        white = mix(white, [0.72, 0.80, 0.58], smoothstep(0.14, 0.0, np.minimum(ph, 1 - ph)) * 0.85)  # 잎집 사이 경계 줄
        white = mix(white, [0.86, 0.92, 0.66], smoothstep(0.004, 0.0, r) * 0.55)  # 가운데 어린 잎은 연두빛
        if cat == "green_onion_v3":  # 바깥 잎집일수록 흰 부분 끝에서 연두가 먼저 든다
            white = mix(white, [0.80, 0.88, 0.62], smoothstep(0.006, 0.0095, r) * smoothstep(x_w - 0.004, x_g, x) * 0.6)
        # 잎 벽 단면: 바깥 껍질 쪽 짙은 청록 → 가운데 초록 → 속 면 연두빛 흰색(벽: 바깥 잎 8.5~10 mm, 속 잎 4.0~5.3 mm)
        g_dark, g_mid, g_pale = np.asarray([0.13, 0.33, 0.15]), np.asarray([0.33, 0.56, 0.27]), np.asarray([0.72, 0.82, 0.60])
        def wall(r0, r1):  # 0 = 속 면, 1 = 바깥 면
            return np.clip((r - r0) / (r1 - r0), 0.0, 1.0)
        s = np.where(r > 0.0069, wall(0.0085, 0.0100), wall(0.0040, 0.0053))
        gcol = mix(mix(np.tile(g_pale, (len(R), 1)), g_mid, smoothstep(0.0, 0.22, s)), g_dark, smoothstep(0.72, 0.95, s))
        if cat == "green_onion_v3":
            gcol = mix(gcol, [0.55, 0.70, 0.42], smoothstep(0.0055, 0.0040, r) * 0.5)  # 속 잎은 더 연함
        flesh = mix(white, gcol, g)
        veins = vnoise(R * np.array([0.08, 1.0, 1.0]), 0.0007)
        sk_w = np.asarray([0.93, 0.95, 0.85])[None] * (1 + 0.05 * veins)[:, None]
        if cat == "green_onion_v2":
            g_out, g_in = np.asarray([0.20, 0.45, 0.14]), np.asarray([0.62, 0.78, 0.46])
            sk_g = mix(np.tile(g_in, (len(R), 1)), g_out, smoothstep(0.0090, 0.0097, r)) * (1 + 0.07 * veins)[:, None]
        else:  # 겉: 짙은 청록 + 흰 가루(왁스) 얼룩, 세로 잎맥
            sk_g = np.asarray([0.20, 0.40, 0.22])[None] * (1 + 0.10 * veins)[:, None]
            bloom = smoothstep(0.05, 0.55, vnoise(R * np.array([0.25, 1.0, 1.0]) + 3.3, 0.004))
            sk_g = mix(sk_g, [0.46, 0.60, 0.48], bloom * 0.45)
            sk_w = mix(sk_w, [0.80, 0.88, 0.64], smoothstep(x_w - 0.006, x_w + 0.002, x) * 0.6)  # 흰 부분 끝의 연두
        sk = mix(sk_w, sk_g, g)
    elif cat in ("pork_slab", "pork_slab10"):  # 생 돼지고기 — 근섬유 결(y 방향)·근막 줄, 드문 마블링, +y 긴 가장자리 지방 띠
        # 11-20 저녁 v3(reports/58): v2 는 지방 = 윤곽 고리 × y 방향 12 mm 전환이라 칼질하는 +x 끝 편의 절반이 흰색으로 번졌다(reports/57 5절).
        #   → 지방은 같은 x 의 +y 윤곽에서 안쪽으로 두께 약 5.5 mm 띠(경계 전환 0.3 mm), 살코기는 더 짙게, 근섬유 다발 경계 줄을 더함.
        x, y = R[:, 0], R[:, 1]
        y_out = np.zeros_like(y)
        for _ in range(3):  # 같은 x 의 +y 윤곽(make_food_shapes.pork_slab 의 초타원, 각도마다 다른 rad 를 고정점 반복으로)
            ang = np.arctan2(y_out / 0.026, x / 0.035)
            rad = 1 + 0.05 * np.sin(3 * ang + 0.7) + 0.035 * np.sin(5 * ang + 2.1) + 0.02 * np.sin(7 * ang + 0.3)
            y_out = 0.026 * rad * np.clip(1 - (np.abs(x) / (0.035 * rad)) ** 4, 0.0, None) ** 0.25
        d_y = y_out - y  # +y 윤곽까지 안쪽 거리
        t_fat = 0.0055 + 0.0012 * vnoise(R * np.array([0.3, 1.0, 1.0]) + 0.9, 0.006)
        side = smoothstep(-0.002, 0.003, y)  # 지방 띠는 +y 쪽만
        fat = smoothstep(t_fat + 0.0003, t_fat - 0.0003, d_y) * side
        fib = vnoise(R * np.array([1.0, 0.12, 1.0]), 0.00045)
        lean = np.asarray([0.73, 0.31, 0.32])[None] * (1 + 0.10 * fib + 0.04 * n1)[:, None]
        bundle = np.abs(vnoise(R * np.array([1.0, 0.10, 1.0]) + 5.1, 0.0016))  # 근섬유 다발(2~3 mm) 경계 = 잡음의 0 근처
        lean = mix(lean, [0.86, 0.66, 0.64], smoothstep(0.045, 0.0, bundle) * 0.45)
        marb = smoothstep(0.62, 0.78, vnoise(R * np.array([0.6, 0.15, 0.6]) + 1.7, 0.0018))
        lean = mix(lean, [0.95, 0.88, 0.84], marb * 0.7)
        inner = np.abs(d_y - (0.016 + 0.003 * np.sin(x / 0.011 + 1.3) + 0.0015 * vnoise(R + 7.7, 0.005)))  # 살 속 지방 줄(목살처럼)
        w_in = 0.0002 + 0.0004 * smoothstep(-0.3, 0.6, vnoise(R * np.array([1.0, 0.3, 1.0]) + 3.9, 0.008))  # 굵기 0.2~0.6 mm, 군데군데 끊김
        lean = mix(lean, [0.94, 0.89, 0.82], smoothstep(w_in, w_in * 0.4, inner) * side * smoothstep(0.004, 0.012, y_out) * 0.85)
        fatcol = np.asarray([0.95, 0.91, 0.84])[None] * (1 + 0.03 * vnoise(R + 0.9, 0.002))[:, None]
        seam = np.exp(-((d_y - t_fat) / 0.0004) ** 2) * side  # 지방과 살 사이 얇은 막
        flesh = mix(mix(lean, fatcol, fat), [0.90, 0.76, 0.74], seam * 0.5)
        w_skin = w_skin * 0.0  # 덩어리 겉도 모두 잘린 면
    elif cat == "garlic_clove":  # 11-20 저녁 v3(reports/58): 깐 마늘 — 상아색 겉의 세로 줄, 단면 가운데 노란 싹, 뿌리 쪽 자른 면은 갈색 판
        x = R[:, 0]
        streak = vnoise(R * np.array([0.10, 1.0, 1.0]) + 0.4, 0.0008)
        sk = np.asarray([0.89, 0.84, 0.66])[None] * (1 + 0.05 * streak + 0.02 * n1)[:, None]
        flesh = np.asarray([0.95, 0.91, 0.75])[None] * (1 + 0.015 * n1)[:, None]
        rc = np.hypot((R[:, 1] - 0.0015 * (1 - (x / 0.013) ** 2)) / 0.0065, (R[:, 2] - 0.0035) / 0.0050)  # 단면 가운데(싹 자리)
        flesh = mix(flesh, [0.86, 0.84, 0.55], smoothstep(0.30, 0.12, rc) * 0.75)
        flesh = mix(flesh, [0.90, 0.85, 0.66], smoothstep(0.70, 1.0, rc) * 0.35)  # 겉 가까이 조금 짙게(비쳐 보임)
        root = smoothstep(0.0108, 0.0118, x)
        flesh = mix(flesh, [0.70, 0.60, 0.40], root * 0.8)
        sk = mix(sk, [0.70, 0.60, 0.40], root * 0.8)
    elif cat in ("luncheon_block", "sausage_cased"):  # 다진 고기 + 작은 지방 점
        flesh = mix(flesh, [0.97, 0.88, 0.84], smoothstep(0.70, 0.85, vnoise(R + 4.2, 0.0007)) * 0.7)
        if cat == "luncheon_block":
            w_skin = w_skin * 0.4  # 통조림 겉면은 조금 더 짙을 뿐
        else:
            w_skin = np.maximum(smoothstep(0.30, 0.65, skin), smoothstep(0.0009, 0.0005, depth))  # 껍질 막
    return srgb(mix(flesh, sk, w_skin))


NEW_FOOD_CATS = ("garlic_clove", "onion_half", "green_onion_white", "luncheon_block", "sausage_cased", "pork_loin_block",
                 "green_onion_v2", "pork_slab", "green_onion_v3", "pork_slab10")


def veg_color_fn(cat, x_rest, p, base_skin=None, base_flesh=None):
    """11-12 채소 색 함수(P, depth, skin) → sRGB. 해당 없는 재료면 None."""
    if cat in NEW_FOOD_CATS and base_skin is not None:
        return lambda P, dep, sk: _new_food_colors(cat, P, dep, sk, base_skin, base_flesh)
    if cat == "carrot":
        s = Section(x_rest, p)
        return lambda P, dep, sk: carrot_colors(P, dep, sk, s)
    if cat == "zucchini":
        s = Section(x_rest, p)
        return lambda P, dep, sk: zucchini_colors(P, dep, sk, s)
    if cat in ("potato", "potato_lit"):
        return lambda P, dep, sk: potato_colors(P, dep, sk)
    return None


def rest_depth(x_rest, meta, p):
    """처음 위치에서 식재료 겉면까지 깊이. 메쉬가 있으면 정확한 거리, 없으면 입자 밀도로 만든 부피의 거리 변환."""
    mp = meta.get("food_mesh")
    if mp and Path(mp).exists():
        m = trimesh.load(mp, force="mesh")
        m.apply_translation([0, 0, meta.get("z_lift", 0.0)])
        try:
            import open3d as o3d
        except ImportError:
            o3d = None
    if mp and Path(mp).exists() and o3d is not None:
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(np.asarray(m.vertices, np.float32)),
                         o3d.core.Tensor(np.asarray(m.faces, np.uint32)))
        return np.maximum(-sc.compute_signed_distance(o3d.core.Tensor(x_rest.astype(np.float32))).numpy(), 0.0)
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    h = p / 4
    lo = x_rest.min(0) - 3 * p
    idx = np.round((x_rest - lo) / h).astype(int)
    g = np.zeros(idx.max(0) + 4 * int(p / h))
    np.add.at(g, tuple(idx.T), 1.0)
    g = gaussian_filter(g, p / h * 0.6)
    inside = g > 0.5 * np.percentile(g[tuple(idx.T)], 50)
    return distance_transform_edt(inside)[tuple(idx.T)] * h


def wood_texture(w=1024, h=768, seed=3):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(float)
    warp = sum(a * np.sin(x / s + rng.uniform(0, 6.28)) for a, s in ((5, 420), (2, 130), (0.6, 33)))
    ring = np.sin((y + warp) / 9.0 + 1.5 * np.sin(y / 170.0)) * 0.5 + 0.5
    fine = gaussian_filter1d(rng.normal(size=(h, w)), 0.7, axis=0)
    fine = gaussian_filter1d(fine, 40, axis=1)
    t = np.clip(0.30 * ring ** 4 + 0.6 * fine * 2.5 + 0.35, 0, 1)
    c0, c1 = np.array([0.80, 0.64, 0.45]), np.array([0.70, 0.53, 0.35])
    img = c0[None, None] * (1 - t[..., None]) + c1[None, None] * t[..., None]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def kabsch(A, B):
    """B ≈ A @ R.T + t 인 회전 R, 이동 t."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - ca @ R.T


def torn_off(Xr, P, r_link=0.003, tear=0.005):
    """11-22(reports/61 대파): 조각 입자가 찢겨 멀리 갈라졌으면 가장 큰 덩어리와 따로 움직인 덩어리를 표시(없으면 None).
    입자 전체에 강체 맞춤을 하면 메쉬가 두 덩어리 사이 공중에 그려진다. 자세한 설명은 teammate/og_mesh_render.py torn_off."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pr = cKDTree(P).query_pairs(r_link, output_type="ndarray")
    cl = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(P), len(P))), directed=False)[1]
    sz = np.bincount(cl)
    if len(sz) < 2:
        return None
    c0 = int(np.argmax(sz))
    R, t = kabsch(Xr[cl == c0], P[cl == c0])
    off = np.zeros(len(P), bool)
    for c in range(len(sz)):
        m = cl == c
        if c != c0 and np.linalg.norm((P[m] - (Xr[m] @ R.T + t)).mean(0)) > tear:
            off |= m
    return off if off.any() else None


def fold_torn(V, nb, w, keep, P):
    """11-22: 찢겨 나간 덩어리 쪽 꼭짓점을 남은 입자 중 가장 가까운 것 자리로 접어 넣는다(teammate/og_mesh_render.py fold_torn 과 같음)."""
    wk = (w * keep[nb]).sum(1) / np.maximum(w.sum(1), 1e-12)
    torn = wk < 0.5
    if not torn.any():
        return V
    j = np.where(keep[nb[torn]], np.arange(nb.shape[1])[None, :], nb.shape[1]).min(1)
    has = j < nb.shape[1]
    tgt = np.empty((int(torn.sum()), 3))
    tgt[has] = P[nb[torn][has, j[has]]]
    if (~has).any():
        from scipy.spatial import cKDTree
        kp = np.flatnonzero(keep)
        tgt[~has] = P[kp[cKDTree(P[kp]).query(V[torn][~has])[1]]]
    V = V.copy()
    V[torn] = tgt
    return V


def split_pieces(mesh, marks, x_rest, X, LAB, steps, pre_trims=(), pre_splits=(), settle=False):
    """처음 메쉬를 칼질마다 나눈다. 칼질 k 에서 라벨이 뒤집힌 입자와 안 뒤집힌 입자가 함께 있는 조각만, 그 조각이 칼질
    직전에 놓인 자세를 되돌린 칼질 평면으로 나눈다. returns [{M: manifold, idx: 입자 번호, planes: [(n, d, 단면 바깥 부호)]}]
    11-13 추가(팀원 솔버 실제 모양 실행): pre_trims = 미리 잘라 낸 평면({n, d, sgn}; sgn −1 이면 n·x ≥ d 쪽을 남김, 잘린 면은 단면으로 그림),
    pre_splits = 처음부터 나뉜 조각의 경계 평면({n, d}; 처음 자리로 나눔 — 채썰기 시연의 편).
    11-19 settle=True(팀원 솔버): '뒤집힘'을 칼질 시작 바로 뒤가 아니라 칼질이 끝나 라벨이 자리 잡은 뒤(다음 칼질 직전,
    없으면 마지막 프레임)와 비교한다. 솔버는 칼질 시작에 날 자리 입자층을 앞 조각으로 붙였다가 곧 몸통으로 되돌린다
    (오이 첫 칼질 40입자). 시작 바로 뒤로 비교하면 그 입자들이 앞 조각 몫이 되고, 다음 칼질에서 다시 뒤집혀 앞 조각
    메쉬를 엉뚱한 쪽으로 나눴다 → 첫 조각 메쉬가 몸통에 남은 40입자를 따라가고 진짜 첫 조각은 안 보임(reports/53)."""
    import manifold3d as mf

    M0 = mf.Manifold(mesh=mf.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                                  tri_verts=np.asarray(mesh.faces, np.uint32)))
    planes0 = []
    for tp in pre_trims:
        n_t, d_t, sg = np.asarray(tp["n"], float), float(tp["d"]), float(tp["sgn"])
        A, B = M0.split_by_plane(tuple(float(v) for v in n_t), d_t)
        M0 = A if sg < 0 else B
        planes0.append((n_t, d_t, sg))
    pieces = [dict(M=M0, idx=np.arange(len(x_rest)), planes=planes0)]
    for sp in pre_splits:
        n_r, d = np.asarray(sp["n"], float), float(sp["d"])
        out = []
        for pc in pieces:
            fl = x_rest[pc["idx"]] @ n_r >= d
            if fl.sum() < 20 or (~fl).sum() < 20:
                out.append(pc)
                continue
            A, B = pc["M"].split_by_plane(tuple(float(v) for v in n_r), d)
            out.append(dict(M=A, idx=pc["idx"][fl], planes=pc["planes"] + [(n_r, d, -1.0)]))
            out.append(dict(M=B, idx=pc["idx"][~fl], planes=pc["planes"] + [(n_r, d, 1.0)]))
        pieces = out
    mk_steps = sorted(int(m["step"]) for m in marks)
    for mk in marks:
        fb = int(np.searchsorted(steps, mk["step"], side="left")) - 1  # 칼질 직전 프레임
        fa = min(fb + 1, len(steps) - 1)
        if settle:  # 11-19: 칼질이 끝나 라벨이 자리 잡은 뒤(다음 칼질 직전, 없으면 마지막 프레임)
            later = [s for s in mk_steps if s > int(mk["step"])]
            fa = int(np.searchsorted(steps, later[0], side="left")) - 1 if later else len(steps) - 1
            fa = min(max(fa, fb + 1), len(steps) - 1)
        flipped = LAB[fa] != LAB[fb]
        yaw = np.radians(mk["yaw_deg"])
        n_w = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        p_w = np.array([mk["x_mm"], mk["y_mm"], 0.0]) / 1000.0
        if "nrm" in mk:  # 11-13: 기울어진 칼(팀원 솔버 og_veg_cut.py --knife_roll_deg)의 칼질 평면 = 법선 + 날 끝 점
            n_w = np.asarray(mk["nrm"], float)
            p_w = np.array([mk["x_mm"], mk["y_mm"], mk.get("z_mm", 0.0)]) / 1000.0
        out = []
        for pc in pieces:
            fl = flipped[pc["idx"]]
            if fl.sum() < 20 or (~fl).sum() < 20:
                out.append(pc)
                continue
            R, t = kabsch(x_rest[pc["idx"]], X[fb][pc["idx"]])
            n_r = R.T @ n_w
            d = float(n_r @ (R.T @ (p_w - t)))
            A, B = pc["M"].split_by_plane(tuple(float(v) for v in n_r), d)  # A: n.x >= d = 뒤집힌 쪽
            out.append(dict(M=A, idx=pc["idx"][fl], planes=pc["planes"] + [(n_r, d, -1.0)]))
            out.append(dict(M=B, idx=pc["idx"][~fl], planes=pc["planes"] + [(n_r, d, 1.0)]))
        pieces = out
    return pieces


def boundary_loops(V, F, n, d, tol):
    """껍질 메쉬의 경계선 중 평면(n.x = d) 위에 있는 것을 순서 있는 고리들로."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(E, axis=0, return_counts=True)
    be = uniq[cnt == 1]
    on = np.abs(V @ n - d) < tol
    be = be[on[be].all(1)]
    nb = {}
    for a, b in be:
        nb.setdefault(int(a), []).append(int(b))
        nb.setdefault(int(b), []).append(int(a))
    loops, seen = [], set()
    for s0 in nb:
        if s0 in seen:
            continue
        loop, prev, cur = [s0], None, s0
        seen.add(s0)
        while True:
            nxt = [v for v in nb[cur] if v != prev and (v not in seen or v == s0)]
            if not nxt or nxt[0] == s0:
                break
            prev, cur = cur, nxt[0]
            loop.append(cur)
            seen.add(cur)
        if len(loop) >= 3:
            loops.append(np.array(loop))
    return loops


def cap_loops_from_faces(V, F, on_k, skin_edges, h):
    """11-14: 평면 k 위에 놓인 면(on_k)들의 바깥 경계를 닫힌 고리로. boundary_loops(껍질 테두리만 봄)는 단면끼리 맞닿은
    모서리(쌓은 편의 윗면·칼질 면 사이처럼 껍질이 없는 곳)에서 고리가 끊겨 단면이 빠지거나 반만 그려졌다(reports/38).
    껍질과 안 닿은 모서리는 h 간격으로 점을 넣는다(맞닿은 다른 단면도 같은 식이라 같은 점) — 단면 안 격자점과 이어지게.
    returns (고리 목록(꼭짓점 번호), 늘린 꼭짓점 배열)"""
    Fk = F[on_k]
    E = np.sort(np.concatenate([Fk[:, [0, 1]], Fk[:, [1, 2]], Fk[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(E, axis=0, return_counts=True)
    be = uniq[cnt == 1]
    Vx, nxt, adj = [V], len(V), {}

    def link(a, b):
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    for a, b in be:
        a, b = int(a), int(b)
        L = float(np.linalg.norm(V[a] - V[b]))
        if (a, b) in skin_edges or L <= h:
            link(a, b)
            continue
        k = int(np.ceil(L / h))
        prev = a
        for s_ in range(1, k):
            Vx.append((V[a] + (V[b] - V[a]) * s_ / k)[None])
            link(prev, nxt)
            prev, nxt = nxt, nxt + 1
        link(prev, b)
    Vext = np.concatenate(Vx)
    loops, seen = [], set()
    for s0 in adj:
        if s0 in seen:
            continue
        loop, prev, cur = [s0], None, s0
        seen.add(s0)
        while True:
            cand = [v for v in adj[cur] if v != prev and (v not in seen or v == s0)]
            if not cand or cand[0] == s0:
                break
            prev, cur = cur, cand[0]
            loop.append(cur)
            seen.add(cur)
        if len(loop) >= 3:
            loops.append(np.array(loop))
    return loops, Vext


def cap_mesh(V, loops, n, sgn, h):
    """평면 위 고리들 안을 경계점 + 간격 h 격자점으로 들로네 삼각분할한 단면. 경계점은 껍질 꼭짓점과 같은 자리.
    returns (꼭짓점, 면, 평면 2D 좌표, (원점, e1, e2), 경계 꼭짓점 번호)"""
    from matplotlib.path import Path as MPath
    from scipy.spatial import Delaunay, cKDTree

    e1 = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    bidx = np.concatenate(loops)
    o = V[bidx].mean(0)
    to2 = lambda P: np.stack([(P - o) @ e1, (P - o) @ e2], 1)
    B2 = to2(V[bidx])
    paths = [MPath(to2(V[l])) for l in loops]
    lo, hi = B2.min(0), B2.max(0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], h), np.arange(lo[1], hi[1], h * 0.866))
    gx = gx + (np.arange(gx.shape[0])[:, None] % 2) * h / 2  # 삼각 격자
    g = np.stack([gx.ravel(), gy.ravel()], 1)
    inside = np.zeros(len(g), bool)
    for pth in paths:
        inside ^= pth.contains_points(g)
    dist, _ = cKDTree(B2).query(g)
    P2 = np.concatenate([B2, g[inside & (dist > 0.55 * h)]])
    tri = Delaunay(P2).simplices
    keep = np.zeros(len(tri), bool)
    cen = P2[tri].mean(1)
    for pth in paths:
        keep ^= pth.contains_points(cen)
    tri = tri[keep]
    # 넓이가 거의 0인 삼각형(경계점이 일직선)은 뺀다. 렌더러가 질감 방향을 계산하다 NaN 을 내 화면 전체가 검게 됐다
    a2 = np.cross(P2[tri[:, 1]] - P2[tri[:, 0]], P2[tri[:, 2]] - P2[tri[:, 0]])
    tri = tri[np.abs(a2) > 1e-3 * h * h]
    P3 = o + P2[:, :1] * e1 + P2[:, 1:] * e2
    P3[: len(bidx)] = V[bidx]
    fn = np.cross(P3[tri[:, 1]] - P3[tri[:, 0]], P3[tri[:, 2]] - P3[tri[:, 0]])
    flip = fn @ (sgn * n) < 0
    tri[flip] = tri[flip][:, ::-1]
    return P3, tri, P2, (o, e1, e2), bidx


def cucumber_bumps(R, sec):
    """껍질 돌기 높이(m): 연한 점 자리의 낮은 혹 + 넓은 굴곡 + 잔물결."""
    _, th = sec.coords(R)
    dd, has = cucumber_warts(R, th)
    return 0.00014 * np.exp(-dd ** 2) * has + 0.00006 * vnoise(R + 2.3, 0.005) + 0.00003 * vnoise(R, 0.0014)


class MeshPiece:
    """조각 하나: 껍질(꼭짓점 색) + 단면들(질감). 매 프레임 입자 움직임으로 꼭짓점을 옮긴다."""

    def __init__(self, idx, parts, x_rest, sigma, k=8):
        from scipy.spatial import cKDTree

        self.idx, self.parts = idx, parts  # parts: [(이름, V0, F, 색 또는 None, uv 또는 None, 질감 또는 None)]
        self.Xr = x_rest[idx]
        V0 = np.concatenate([pt[1] for pt in parts])
        # 11-14 밤: 입자가 k 개보다 적은 작은 조각(예: 27개)은 cKDTree 가 없는 이웃을 번호 n·거리 inf 로 채워
        # 렌더러가 IndexError 로 멈췄다(reports/43 당근 채썰기) → 이웃 수를 입자 수 이하로.
        k = max(1, min(int(k), len(self.Xr)))
        dist, self.nb = cKDTree(self.Xr).query(V0, k=k)
        if k == 1:
            dist, self.nb = dist[:, None], self.nb[:, None]
        self.w = np.exp(-(dist / sigma) ** 2)
        self.V0 = V0
        self.cuts = np.cumsum([0] + [len(pt[1]) for pt in parts])

    def deform(self, Xf, r_max=0.004, resid=1.0):
        """resid: 강체 맞춤에서 남는 변형을 겉면에 옮기는 비율(0 = 강체로만)."""
        P = Xf[self.idx]
        R, t = kabsch(self.Xr, P)
        keep = None
        if len(P) >= 8 and np.mean(np.linalg.norm(P - (self.Xr @ R.T + t), axis=1) >= r_max) > 0.1:
            off = torn_off(self.Xr, P)  # 11-22: 찢겨 멀리 갈라진 덩어리는 맞춤에서 뺌(og_mesh_render.torn_off 와 같음)
            if off is not None:
                keep = ~off
                R, t = kabsch(self.Xr[keep], P[keep])
        if resid <= 0:
            V = self.V0 @ R.T + t
            if keep is not None:
                V = fold_torn(V, self.nb, self.w, keep, P)
            return [V[a:b] for a, b in zip(self.cuts[:-1], self.cuts[1:])]
        res = P - (self.Xr @ R.T + t)
        ok = np.linalg.norm(res, axis=1) < r_max  # 떨어져 나간 입자 몇 개가 메쉬를 잡아끌지 않게
        if keep is not None:
            ok &= keep
        w = self.w * ok[self.nb]
        V = self.V0 @ R.T + t + resid * (w[..., None] * res[self.nb]).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-9)
        if keep is not None:
            V = fold_torn(V, self.nb, self.w, keep, P)
        return [V[a:b] for a, b in zip(self.cuts[:-1], self.cuts[1:])]


def build_mesh_pieces(meta, x_rest, X, LAB, steps, sec, base_skin, base_flesh, p, tex_px=1024, veg=None):
    mesh = trimesh.load(meta["food_mesh"], force="mesh")
    if meta.get("food_xform") is not None:  # 11-13: 팀원 솔버 실제 모양(veg_shape_prep.py) — 원래 메쉬를 그 좌표로
        mesh.apply_transform(np.asarray(meta["food_xform"], float))
    mesh.apply_translation([0, 0, meta.get("z_lift", 0.0)])
    if mesh.edges_unique_length.mean() > 0.0006:
        mesh = mesh.subdivide()
    # 사진에서 만든 형상은 복셀을 메쉬로 바꾼 것이라 0.5mm 계단이 있다. 모양은 두고 계단만 지운다
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=40)
    try:
        import open3d as o3d
    except ImportError:
        o3d = None
    if o3d is not None:  # 겉면까지 정확한 깊이
        sdf = o3d.t.geometry.RaycastingScene()
        sdf.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                          o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
        depth_of = lambda P: np.maximum(-sdf.compute_signed_distance(o3d.core.Tensor(P.astype(np.float32))).numpy(), 0.0)
    else:  # Open3D 가 없으면: 겉면에서 고르게 뽑은 점까지 가장 가까운 거리(무늬 색에만 쓰여 0.1 mm 안팎 차이는 안 보임)
        from scipy.spatial import cKDTree

        _surf = cKDTree(trimesh.sample.sample_surface_even(mesh, 200000, seed=0)[0])
        depth_of = lambda P: _surf.query(np.asarray(P, float))[0]
    color = (lambda P, dep, sk: cucumber_colors(P, dep, sk, sec, base_skin, base_flesh)) if sec is not None else (
        lambda P, dep, sk: generic_colors(P, dep, sk, base_skin, base_flesh))
    to_src = lambda P: P  # 11-14: 눕혀 쌓은 편(color_in_source_frame) — 무늬는 원래 채소 좌표(길이 축 x)에서 잰다
    if meta.get("color_in_source_frame") and meta.get("food_xform") is not None:
        Ti = np.linalg.inv(np.asarray(meta["food_xform"], float))
        to_src = lambda P: P @ Ti[:3, :3].T + Ti[:3, 3]
    if veg is not None:  # 11-12 당근·애호박·감자(render_surface main 이 category 를 넘길 때만). 11-14: 단면 표도 원래 좌표 입자로
        color = veg_color_fn(veg, to_src(x_rest), p, base_skin, base_flesh) or color  # 11-20: 새 식재료는 기본 색도 넘김
    if meta.get("color_in_source_frame") and meta.get("food_xform") is not None:
        color0 = color
        color = lambda P, dep, sk: color0(to_src(P), dep, sk)
    tol = 2e-6
    out = []
    settle = str(meta.get("solver", "")).startswith("teammate Warp MPM")  # 11-19: Genesis 실행은 확인 안 해서 그대로
    for pi, pc in enumerate(split_pieces(mesh, meta.get("marks", []), x_rest, X, LAB, steps, meta.get("pre_trims", []),
                                         meta.get("pre_splits", []), settle=settle)):
        m = pc["M"].to_mesh()
        V = np.asarray(m.vert_properties, float)[:, :3]
        F = np.asarray(m.tri_verts, np.int64)
        on_cap = np.zeros(len(F), bool)
        for n, d, _ in pc["planes"]:
            on_cap |= (np.abs(V @ n - d) < tol)[F].all(1)
        Fs = F[~on_cap]
        used = np.unique(Fs)
        remap = -np.ones(len(V), np.int64)
        remap[used] = np.arange(len(used))
        Vs, Fs = V[used], remap[Fs]
        # 껍질: 처음 겉면 위 점이라 깊이 0. 단면 경계와 먼 곳만 돌기만큼 바깥으로 민다
        Cs = color(Vs, np.zeros(len(Vs)), np.ones(len(Vs)))
        if sec is not None:
            nrm = trimesh.Trimesh(Vs, Fs, process=False).vertex_normals
            near_cut = np.zeros(len(Vs), bool)
            for n, d, _ in pc["planes"]:
                near_cut |= np.abs(Vs @ n - d) < 0.0006
            Vs = Vs + nrm * (cucumber_bumps(to_src(Vs), sec) * ~near_cut)[:, None]
        parts = [(f"p{pi}_skin", Vs, Fs, Cs, None, None)]
        Fsk = F[~on_cap]  # 11-14: 단면 고리는 메쉬의 평면 면에서(cap_loops_from_faces). 껍질과 맞닿은 모서리는 그대로
        skin_e = set(map(tuple, np.sort(np.concatenate([Fsk[:, [0, 1]], Fsk[:, [1, 2]], Fsk[:, [2, 0]]]), axis=1).tolist()))
        for j, (n, d, sgn) in enumerate(pc["planes"]):
            on_k = (np.abs(V @ n - d) < tol)[F].all(1)
            if not on_k.any():
                continue
            loops, Vext = cap_loops_from_faces(V, F, on_k, skin_e, 0.0008)
            if not loops:
                continue
            P3, tri, P2, (o, e1, e2), _ = cap_mesh(Vext, loops, n, sgn, 0.0008)
            lo = P2.min(0)
            S = float((P2.max(0) - lo).max()) + 1e-4
            uv = (P2 - lo) / S
            # 질감: 화소 중심의 처음 위치에서 과육·씨·껍질 테두리 색
            g = (np.arange(tex_px) + 0.5) / tex_px
            U, Vv = np.meshgrid(g, g)
            Pt = o + (lo[0] + U.ravel()[:, None] * S) * e1 + (lo[1] + Vv.ravel()[:, None] * S) * e2
            img = color(Pt, depth_of(Pt), np.zeros(len(Pt))).reshape(tex_px, tex_px, 3)
            img = (np.clip(img, 0, 1) ** (1 / 1.0) * 255).astype(np.uint8)[::-1].copy()  # 질감 v 는 아래가 0
            parts.append((f"p{pi}_cap{j}", P3, tri, None, uv[tri].reshape(-1, 2), img))
        # 남는 변형은 5mm 범위로 고르게 편다(8개 이웃이면 칼에 밀린 격자 잡음 0.3~0.7mm 가 껍질에 혹처럼 남았다)
        out.append(MeshPiece(pc["idx"], parts, x_rest, sigma=2 * p, k=32))
    return out
