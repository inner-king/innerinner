"""로봇 데이터 생성 화면의 '현실 장면'·겉모습 무작위화(food_look.py)가 쓸 무늬 그림을 만든다 (2026-11-26, reports/81).

cutsim 환경에서 한 번 돌리면 data/looks/textures/*.png 와 index.json 이 생긴다(시드 고정 — 같은 그림이 다시 나옴).
  python scripts/cutting/food/make_look_textures.py
무늬 = 도마(흰 플라스틱·나무·대나무·회색 플라스틱), 조리대(밝은 화강암·대리석·나무·어두운 단색), 벽(흰 타일·베이지 타일·칠한 벽).
'현실 장면' 기본값은 팀원 10-25 주방 녹화(흰 플라스틱 도마, 밝은 회색 조리대, 흰 타일 — 사진에서 눈으로 고름)에 맞춘다.
그림 크기는 실제 크기 기준(size_m)으로 적어 두고, food_look.py 가 월드 투영으로 붙일 때 이 크기로 늘인다.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[3]  # 저장소 뿌리
sys.path.insert(0, str(Path(__file__).resolve().parent))
import food_mesh_lib as RS  # noqa: E402  (나무결)

OUT = Path(os.environ.get("FOOD_LOOK_TEXTURES", ROOT / "Data" / "food" / "looks" / "textures"))


def _n(rng, h, w, s):
    a = gaussian_filter(rng.normal(size=(h, w)), s)
    return (a - a.min()) / (np.ptp(a) + 1e-9)


def plastic(rng, w, h, c, scratches=900, dark=0.06):
    """도마 플라스틱: 고른 색 + 아주 옅은 얼룩 + 칼자국(가는 직선, 대부분 한 방향)"""
    img = np.ones((h, w, 3)) * np.array(c)
    img *= (0.975 + 0.035 * _n(rng, h, w, 24))[..., None]
    img *= (0.99 + 0.012 * _n(rng, h, w, 1.2))[..., None]
    m = np.zeros((h, w))
    for _ in range(scratches):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.normal(0.0, 0.35) + (np.pi / 2 if rng.random() < 0.3 else 0.0)
        L = rng.uniform(15, 140)
        t = np.linspace(0, L, int(L))
        xs = np.clip((x0 + t * np.cos(ang)).astype(int), 0, w - 1)
        ys = np.clip((y0 + t * np.sin(ang)).astype(int), 0, h - 1)
        m[ys, xs] = np.maximum(m[ys, xs], rng.uniform(0.3, 1.0))
    m = gaussian_filter(m, 0.6)
    img *= (1.0 - dark * np.clip(m * 2.0, 0, 1))[..., None]
    return img


def granite(rng, w, h, c, c_dark, c_light, dens=0.10):
    """화강암: 바탕 + 어두운·밝은 점(크기 여러 가지)"""
    img = np.ones((h, w, 3)) * np.array(c)
    img *= (0.95 + 0.08 * _n(rng, h, w, 30))[..., None]
    for col, d, s in ((c_dark, dens, 0.8), (c_light, dens * 0.6, 0.9), (c_dark, dens * 0.15, 2.0)):
        spots = gaussian_filter((rng.random((h, w)) < d * 0.08).astype(float), s)
        a = np.clip(spots / (spots.max() + 1e-9) * 1.6, 0, 1)[..., None]
        img = img * (1 - a) + np.array(col) * a
    return img


def marble(rng, w, h, c1, c2):
    n = _n(rng, h, w, 18)
    y, x = np.mgrid[0:h, 0:w].astype(float)
    vein = np.abs(np.sin((x * 0.6 + y * 0.8) / 40.0 + 6.0 * n)) ** 0.35
    t = (0.7 * vein + 0.3 * _n(rng, h, w, 3))[..., None]
    return np.array(c1) * t + np.array(c2) * (1 - t)


def tiles(rng, w, h, size_m, c, tile_m, grout_c, grout_m=0.003, tile_h=None):
    img = np.ones((h, w, 3)) * np.array(c)
    ppm = w / size_m[0]
    tw, th_ = tile_m * ppm, (tile_h or tile_m) * ppm
    y, x = np.mgrid[0:h, 0:w].astype(float)
    ti, tj = (x // tw).astype(int), (y // th_).astype(int)
    jit = 1.0 + 0.035 * rng.normal(size=(ti.max() + 2, tj.max() + 2))[ti, tj]
    img *= jit[..., None]
    img *= (0.97 + 0.04 * _n(rng, h, w, 2))[..., None]
    g = max(1.0, grout_m * ppm)
    grout = (np.mod(x, tw) < g) | (np.mod(y, th_) < g)
    img[grout] = np.array(grout_c)
    return img


def bamboo(rng, w, h, c):
    """대나무 도마: 좁은 세로 띠(색 조금씩 다름) + 마디 무늬"""
    img = np.ones((h, w, 3)) * np.array(c)
    strip = int(w / 22)
    for i in range(0, w, strip):
        img[:, i:i + strip] *= rng.uniform(0.9, 1.06)
        img[:, i:i + 1] *= 0.85
    img *= (0.96 + 0.06 * _n(rng, h, w, (6, 1.5)))[..., None]
    for _ in range(14):
        yy, xx = int(rng.uniform(0, h)), int(rng.uniform(0, w // strip)) * strip
        img[max(0, yy - 2):yy + 2, xx:xx + strip] *= 0.82
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261126)
    T = {}  # 이름 → (그림, 실제 크기 [m], 종류, 설명)

    def save(name, img, size_m, kind, note):
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        im.save(OUT / f"{name}.png")
        T[name] = {"file": f"{name}.png", "size_m": list(size_m), "kind": kind, "note": note}

    # 도마 (실제 도마 0.34 × 0.24 m 에 한 장)
    save("board_white_plastic", plastic(rng, 1024, 724, [0.93, 0.93, 0.915]), (0.34, 0.24), "board", "흰 플라스틱(팀원 주방 녹화)")
    save("board_gray_plastic", plastic(rng, 1024, 724, [0.62, 0.64, 0.64], dark=0.08), (0.34, 0.24), "board", "회색 플라스틱")
    wood = RS.wood_texture(1024, 724, seed=7).astype(float) / 255.0
    save("board_wood", wood * np.array([1.05, 0.98, 0.88]), (0.34, 0.24), "board", "나무(밝은 결)")
    save("board_bamboo", bamboo(rng, 1024, 724, [0.80, 0.66, 0.44]), (0.34, 0.24), "board", "대나무")
    # 조리대 (1.6 × 1.2 m 한 장 — 이음매가 덜 보이게 크게)
    save("counter_granite_light", granite(rng, 1600, 1200, [0.80, 0.80, 0.78], [0.42, 0.42, 0.42], [0.95, 0.95, 0.94]),
         (1.6, 1.2), "counter", "밝은 회색 화강암(팀원 주방 조리대에 가깝게)")
    save("counter_marble", marble(rng, 1600, 1200, [0.93, 0.93, 0.92], [0.70, 0.70, 0.72]), (1.6, 1.2), "counter", "흰 대리석")
    save("counter_wood", RS.wood_texture(1600, 1200, seed=11).astype(float) / 255.0 * np.array([0.95, 0.86, 0.74]), (1.6, 1.2),
         "counter", "나무 조리대")
    save("counter_dark", granite(rng, 1600, 1200, [0.22, 0.22, 0.23], [0.10, 0.10, 0.10], [0.45, 0.45, 0.46], dens=0.06),
         (1.6, 1.2), "counter", "어두운 돌")
    # 벽 (2.4 × 1.2 m)
    save("wall_tile_white", tiles(rng, 2400, 1200, (2.4, 1.2), [0.93, 0.93, 0.92], 0.15, [0.72, 0.72, 0.70]), (2.4, 1.2), "wall",
         "흰 정사각 타일 15 cm")
    save("wall_subway_beige", tiles(rng, 2400, 1200, (2.4, 1.2), [0.88, 0.84, 0.76], 0.15, [0.66, 0.64, 0.60], tile_h=0.075),
         (2.4, 1.2), "wall", "베이지 벽돌꼴 타일")
    save("wall_painted", np.ones((600, 1200, 3)) * np.array([0.86, 0.85, 0.80]) * (0.96 + 0.05 * _n(rng, 600, 1200, 8))[..., None],
         (2.4, 1.2), "wall", "칠한 벽")
    (OUT / "index.json").write_text(json.dumps(T, ensure_ascii=False, indent=1))
    print(f"{len(T)} 장 → {OUT}")


if __name__ == "__main__":
    main()
