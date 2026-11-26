"""식재료 안의 '부위'(질긴 조직 자리)를 모양 좌표로 정한다 (2026-11-21, reports/60).

numpy 만 쓴다 — 솔버 실행기(scripts/cutting/food/og_veg_cut.py, simfoundry 환경)와 그림 스크립트(cutsim 환경)가 같이 부른다.

pork_tissue(P) — 생 돼지 덩이(make_food_shapes.pork_slab·pork_slab10)의 질긴 조직:
  사용자 11-21: "고기는 … 칼이 끝까지 내려간 상태에서도 앞뒤로 칼을 움직여야 끊어지는 부위도 있는데" → "일부 부위에만 걸어줘".
  실제 목살·등심에서 칼이 잘 안 끊는 곳은 결합 조직(근막·힘줄·지방을 싼 막)이다. 렌더(scripts/render_surface.py 의 pork_slab 색)가
  이미 그리는 두 자리를 그대로 질긴 조직으로 둔다(잡음 없이 가운데 값):
    ① +y 긴 가장자리 지방 띠와 그 안쪽 지방–살 사이 막: +y 윤곽에서 안쪽 6.5 mm 까지(지방 5.5 mm + 막 약 1 mm)
    ② 살 속 근막 줄(목살처럼 근육 사이 흰 줄): +y 윤곽에서 안쪽 d_in(x) = 16 + 3·sin(x/11 mm + 1.3) mm 에 너비 2.5 mm
  둘 다 +y 쪽 절반에만 있다(렌더와 같음). 덩이 두께 방향(z)으로는 위아래 통째(기둥).
  가정: 위치·너비는 렌더 무늬에 맞춘 것이지 실측이 아니다. 질긴 정도(필요한 미끄러짐 40 mm)는 두 자리 같음.
"""
import numpy as np


def _pork_y_out(x):
    """같은 x 의 +y 윤곽(make_food_shapes.pork_slab 의 초타원, 각도마다 다른 rad 를 고정점 반복으로 — render_surface 와 같은 식)."""
    y_out = np.zeros_like(x)
    for _ in range(3):
        ang = np.arctan2(y_out / 0.026, x / 0.035)
        rad = 1 + 0.05 * np.sin(3 * ang + 0.7) + 0.035 * np.sin(5 * ang + 2.1) + 0.02 * np.sin(7 * ang + 0.3)
        y_out = 0.026 * rad * np.clip(1 - (np.abs(x) / (0.035 * rad)) ** 4, 0.0, None) ** 0.25
    return y_out


FAT_SEAM_W = 0.0065   # ① 지방 띠 + 막 [m]
INNER_HALF_W = 0.00125  # ② 근막 줄 반너비 [m]


def pork_tissue_parts(P):
    """(① 지방 띠·막, ② 살 속 근막 줄) 불 배열 둘. P (N,3) 모양 좌표 [m]."""
    P = np.asarray(P, float)
    x, y = P[:, 0], P[:, 1]
    y_out = _pork_y_out(x)
    d_y = y_out - y
    side = y > 0.0005  # 렌더 side = smoothstep(-2, 3 mm, y) 의 가운데
    fat = side & (d_y < FAT_SEAM_W)
    d_in = 0.016 + 0.003 * np.sin(x / 0.011 + 1.3)
    inner = side & (y_out > 0.008) & (np.abs(d_y - d_in) < INNER_HALF_W)
    return fat, inner


def pork_tissue(P):
    fat, inner = pork_tissue_parts(P)
    return fat | inner


REGIONS = {"pork_tissue": pork_tissue}
