"""팀원 Warp MPM 솔버에 우리 보정을 붙이는 감싸개 (2026-11-13, reports/34 6·7 절에서 시험한 것을 떼어 냄).

- 손 'press': 손 상자 안 입자가 위로 들리거나 옆으로 밀리지 않게만 한다(아래로는 내려앉고 변형은 그대로 계산). 'pin' 은 위치를 얼림.
- 칼 옆 조각 감쇠: 칼날 면에서 sep_r 안에 입자가 있는 조각(과 맞닿은 조각)의 속도·C 를 단계마다 1/(1 + c dt) 배 —
  우리 Genesis `--sep_damp 300 --sep_damp_r 0.015 --sep_damp_s 0.06` 과 같은 규칙. 칼질(cut 단계) 처음부터 켜고, 날 끝이 재료 윗면 위로
  나간 뒤 sep_after 초에 끈다(팀원 솔버는 '끝까지 잘림' 직전부터 조각이 밀려 나가므로 잘린 순간부터 켜면 늦다).
- 칼 접착(물막, 11-13 추가): 우리 Genesis 패치 7단계 v3 와 같은 꼴. 칼날 옆면(날 끝 위, 등 아래)에서 r1 안 입자에
  흡인 p_a(칼 면 쪽으로 당김), 떼기·미끄럼 정적 한도 σ·τ, 떼기·미끄럼 점성 c_n·c_t 를 건다(칼과의 상대 속도 기준, 암시적이라 안정).
  Genesis 에서 가장 나았던 판(④: σ 0, τ 0, c_n 5e4, c_t 5e3 Pa·s/m, p_a 1 kPa, 거리 0.83~5 mm)이 기본값.
  팀원 솔버에서는 복사본의 '칼날 너머 조각끼리 안 맞닿음'(sim.blade_split = 1, patches/make_warp_blade_split.py)을 같이 켜야 조각이 칼 면에 닿는다.
  adh_release="top"(기본): 칼이 맨 위에서 멈추면 접착을 꺼 붙은 조각을 떨어뜨린다(규칙).
  팀원 솔버의 칼은 다가오는 재료를 밀어낼 뿐 멀어지는 재료를 붙잡지 않아 원래는 조각이 칼에 붙지 않는다.
- 도마 접촉 고침은 솔버 복사본(`patches/make_warp_board_contact.py` → `<빠른3 솔버 복사본>`)에 있고, 여기서는 문턱(sim.board_tol)만 정한다.

이 모듈은 Isaac Sim(og.launch) 을 띄운 **뒤에** import 한다: Warp 커널을 그 프로세스의 Warp(Kit 의 omni.warp.core)로 만들어야 한다.
좌표: '우리 좌표' = 재료 상자 가운데가 x·y 0, 도마 면이 z 0 (Genesis 실행과 같음). 솔버 좌표 = 우리 좌표 + off.
"""
import math

import numpy as np
import warp as wp

from simfoundry.cutting.mpm import KnifeState, _edge_w, _half_thickness, _spine_w

ADHESION_FILM = dict(sigma=0.0, tau=0.0, c_n=5.0e4, c_t=5.0e3, p_a=1000.0, r0=0.83e-3, r1=5.0e-3, min_kw=0.5e-3)
# 11-13 맞춤(reports/36): 물막은 맞닿은 면에만(접착 구역 = 칼 면 접촉 거리 + 1.5 입자층), 버티는 힘은 물막 정지 미끄럼 저항 τ 가 주로 맡는다
# (면적 비례 → 무게(두께 비례)와 겨뤄 얇은 조각만 붙음). 흡인은 칼 면에 닿아 있을 정도로만. '보통 수분' = 이 값 × wet 1.
# 오이 40 mm·격자 1.67 mm 에서 3 mm 딱 붙음, 6 mm 안 올라옴(tune/tau30_t3·t6). 수분 많음 = wet 2.5 등(값은 실측 없는 추정).
ADHESION_FILM2 = dict(sigma=0.0, tau=30.0, c_n=7.5e3, c_t=750.0, p_a=150.0, r0=0.83e-3, r1=5.0e-3, min_kw=0.5e-3, layers=1.5)
# 11-13 저녁 정정: film2 는 칼날이 지나간 자리에 남은(칼날 속) 입자가 칼에 강하게 묶여 붙던 것이었다. 칼 접촉 쪽 판정을 고쳐
# (make_warp_knife_side.py: 칼날 속 입자는 조각 쪽으로 셈) 조각이 칼 면 밖(약 0.9 mm)에 놓이자 film2 로는 3 mm 도 안 붙었다.
# film3 = 고친 접촉에서 다시 맞춘 값(reports/36 README 2-3 절).
# 얇은 조각은 칼을 들기 시작하면 칼 면에서 벗겨져서 흡인(p_a)·떼기 점성(c_n)이 어느 정도 있어야 한다. 3 mm 딱 붙음·6 mm 3 mm 들렸다 떨어짐
# (tune/f3C_t3·t6). 흡인을 600 Pa 로 올리면 칼 마찰 × 흡인 때문에 6 mm 도 붙는다(f3B). '보통 수분' = 이 값 × wet 1.
ADHESION_FILM3 = dict(sigma=0.0, tau=30.0, c_n=2.0e4, c_t=750.0, p_a=250.0, r0=0.83e-3, r1=5.0e-3, min_kw=0.5e-3, layers=2.0)


@wp.kernel
def _adh_gap_hist(x: wp.array(dtype=wp.vec3), piece: wp.array(dtype=int), knife: KnifeState, r_scan: float, min_kw: float,
                  d_lo: float, inv_bw: float, nb: int, hist: wp.array(dtype=int)):
    """11-17(reports/47) 조각 면 맞춤 1단계: _adhere 와 같은 칼날 옆면 높이 범위에서 칼 면까지 거리 d ≤ r_scan 인 입자를
    조각·칼의 쪽(±e_v)마다 d 막대그림(칸 너비 1/inv_bw, d_lo 부터 nb 칸)에 센다."""
    p = wp.tid()
    if knife.enabled == 0:
        return
    r = x[p] - knife.origin
    ku = wp.dot(r, knife.e_u)
    if wp.abs(ku) > knife.half_length:
        return
    e = _edge_w(knife, ku)
    kw = wp.dot(r, knife.e_w) - e
    if kw < min_kw or kw > _spine_w(knife, ku) - e:
        return
    kv = wp.dot(r, knife.e_v)
    d = wp.abs(kv) - _half_thickness(knife, kw)
    if d > r_scan:
        return
    side = int(0)
    if kv > 0.0:
        side = 1
    b = wp.clamp(int(wp.floor((d - d_lo) * inv_bw)), 0, nb - 1)
    wp.atomic_add(hist, (piece[p] * 2 + side) * nb + b, 1)


@wp.kernel
def _adh_gap_pct(hist: wp.array(dtype=int), nb: int, q: float, d_lo: float, bw: float, g_lo: float, g_hi: float, min_n: int,
                 shift: wp.array(dtype=float)):
    """조각 면 맞춤 2단계: 조각·쪽마다 막대그림의 q 백분위 거리 = 그 조각 칼 쪽 면이 칼 면에서 떨어진 정도(틈, 음수면 칼날 속으로 듦).
    g_lo~g_hi 로 자른 값을 shift 에 둔다(입자가 min_n 보다 적으면 0). _adhere 는 d 대신 d − shift 로 구역·흡인을 본다."""
    k = wp.tid()
    tot = int(0)
    for b in range(nb):
        tot = tot + hist[k * nb + b]
    s = float(0.0)
    if tot >= min_n:
        target = q * float(tot)
        c = int(0)
        g = d_lo + float(nb) * bw
        found = int(0)
        for b in range(nb):
            c = c + hist[k * nb + b]
            if found == 0 and float(c) >= target:
                g = d_lo + (float(b) + 0.5) * bw
                found = 1
        s = wp.clamp(g, g_lo, g_hi)
    shift[k] = s


@wp.kernel
def _adhere(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), knife: KnifeState, dt: float, rho: float, r0: float,
            r1: float, sigma: float, tau: float, c_n: float, c_t: float, p_a: float, min_kw: float, d_c: float,
            stats: wp.array(dtype=float), piece: wp.array(dtype=int), shift: wp.array(dtype=float), anchor: int):
    """칼날 옆면 접착(물막). 날 끝 아래(kw < min_kw)·등 위·칼날 길이 밖은 뺀다(날 끝이 갈고리처럼 재료를 끄는 것 막기, Genesis 11-12 교훈).
    11-13 고침: 흡인은 칼 면까지 거리 d > d_c(입자 반지름, 아직 안 닿음)일 때만, 떼기 점성은 d > 0(칼날 속에 안 든 입자)일 때만.
    전에는 이미 닿은 입자를 칼날 중심 쪽으로 계속 당겨, 격자점에서만 막는 팀원 칼 접촉을 지나 칼날 속(최대 1.2 mm)으로 들어갔고
    그 입자들은 칼 면을 누르지 못해 마찰로 버티지 못했다(reports/36 tune/t3_w05).
    11-17 anchor=1(reports/47): d 를 칼 면이 아니라 그 조각 자신의 칼 쪽 면(_adh_gap_pct 의 shift)에서 잰다. 썬 뒤 갈라진 면은
    칼날 실제 두께 안 격자점에서만 칼에 닿아서, 칼 평면이 격자 칸 안 어디에 있느냐에 따라 조각이 칼 면에서 −0.2~+0.4 mm 떨어져 멈춘다(가짜 틈).
    구역이 칼 면에 묶여 있으면 그 틈만큼 구역에 드는 입자층이 줄어 τ 가 최대 약 25 % 약해졌다."""
    p = wp.tid()
    if knife.enabled == 0:
        return
    r = x[p] - knife.origin
    ku = wp.dot(r, knife.e_u)
    if wp.abs(ku) > knife.half_length:
        return
    e = _edge_w(knife, ku)
    kw = wp.dot(r, knife.e_w) - e
    if kw < min_kw or kw > _spine_w(knife, ku) - e:
        return
    kv = wp.dot(r, knife.e_v)
    d = wp.abs(kv) - _half_thickness(knife, kw)
    if anchor == 1:
        side = int(0)
        if kv > 0.0:
            side = 1
        d = d - shift[piece[p] * 2 + side]
    if d > r1:
        return
    w = wp.clamp((r1 - d) / (r1 - r0), 0.0, 1.0)
    n = knife.e_v
    if kv < 0.0:
        n = -knife.e_v
    vk = knife.lin_vel + wp.cross(knife.ang_vel, r)
    rv = v[p] - vk
    k = w * dt / (rho * 0.5 * (r0 + r1))
    vn = wp.dot(rv, n)
    if vn > 0.0 and d > 0.0:
        vn_new = wp.max(vn - sigma * k, 0.0) / (1.0 + c_n * k)
        rv = rv + (vn_new - vn) * n
    rt = rv - wp.dot(rv, n) * n
    rtl = wp.length(rt)
    if rtl > 1.0e-12:
        rt_new = wp.max(rtl - tau * k, 0.0) / (1.0 + c_t * k)
        rv = rv - rt * (1.0 - rt_new / rtl)
    if d > d_c:
        rv = rv - p_a * k * n
    v_new = vk + rv
    wp.atomic_add(stats, 0, 1.0)
    wp.atomic_add(stats, 1, v_new[2] - v[p][2])
    v[p] = v_new


@wp.func
def _in_adh_zone(xp: wp.vec3, knife: KnifeState, r1: float, min_kw: float):
    """_adhere 와 같은 접착 구역(칼날 옆면, 날 끝 위 min_kw ~ 등 아래, 칼 면에서 r1 안)."""
    inside = int(0)
    if knife.enabled == 1:
        r = xp - knife.origin
        ku = wp.dot(r, knife.e_u)
        if wp.abs(ku) <= knife.half_length:
            e = _edge_w(knife, ku)
            kw = wp.dot(r, knife.e_w) - e
            if kw >= min_kw and kw <= _spine_w(knife, ku) - e:
                if wp.abs(wp.dot(r, knife.e_v)) - _half_thickness(knife, kw) <= r1:
                    inside = 1
    return inside


@wp.kernel
def _adh_flag(x: wp.array(dtype=wp.vec3), piece: wp.array(dtype=int), knife: KnifeState, r1: float, min_kw: float,
              flag: wp.array(dtype=int)):
    p = wp.tid()
    if _in_adh_zone(x[p], knife, r1, min_kw) == 1:
        wp.atomic_max(flag, piece[p], 1)


@wp.kernel
def _board_flag(x: wp.array(dtype=wp.vec3), piece: wp.array(dtype=int), board_z: float, h: float, touch: wp.array(dtype=int)):
    p = wp.tid()
    if x[p][2] - board_z < h:
        wp.atomic_max(touch, piece[p], 1)


@wp.kernel
def _bounce(v: wp.array(dtype=wp.vec3), piece: wp.array(dtype=int), flag: wp.array(dtype=int), touch: wp.array(dtype=int),
            s: float):
    """도마에 닿은(touch=1) 조각 전체의 위쪽 속도를 s 배(칼 접착 구역에 든 조각 flag=1 은 뺌) — 떨어진 조각이 고무공처럼 튀는 것 줄이기.
    (11-13 첫 판은 도마 면 1.5칸 안 입자만 줄였는데 조각 몸 전체가 눌렸다 펴지며 튀어 효과가 없었다: 58 → 55 mm.)"""
    p = wp.tid()
    f = piece[p]
    if touch[f] == 1 and flag[f] == 0:
        vel = v[p]
        if vel[2] > 0.0:
            v[p] = wp.vec3(vel[0], vel[1], vel[2] * s)


@wp.kernel
def _piece_knife_stats(x: wp.array(dtype=wp.vec3), piece: wp.array(dtype=int), knife: KnifeState,
                       cnt: wp.array(dtype=float), suvw: wp.array(dtype=wp.vec3), wmin: wp.array(dtype=float),
                       nbelow: wp.array(dtype=float)):
    """11-19 칼등 위 넘어뜨리기: 조각마다 입자 수, 칼 좌표(u 칼날 길이·v 칼 평면 법선·w 날 끝→등) 합, 가장 낮은 w, 칼등보다 낮은 입자 수."""
    p = wp.tid()
    f = piece[p]
    r = x[p] - knife.origin
    u = wp.dot(r, knife.e_u)
    w = wp.dot(r, knife.e_w)
    wp.atomic_add(cnt, f, 1.0)
    wp.atomic_add(suvw, f, wp.vec3(u, wp.dot(r, knife.e_v), w))
    wp.atomic_min(wmin, f, w)
    if w < _spine_w(knife, u):
        wp.atomic_add(nbelow, f, 1.0)


@wp.kernel
def _topple_kick(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), C: wp.array(dtype=wp.mat33),
                 piece: wp.array(dtype=int), omega: wp.array(dtype=wp.vec3), pivot: wp.array(dtype=wp.vec3)):
    """조각 전체에 강체 회전 속도 omega × (x - pivot) 를 더한다(APIC 의 C 에도 같은 회전을 더함). omega = 0 인 조각은 그대로."""
    p = wp.tid()
    f = piece[p]
    om = omega[f]
    if wp.length(om) > 0.0:
        v[p] = v[p] + wp.cross(om, x[p] - pivot[f])
        C[p] = C[p] + wp.mat33(0.0, -om[2], om[1], om[2], 0.0, -om[0], -om[1], om[0], 0.0)


@wp.kernel
def _press(v: wp.array(dtype=wp.vec3), idx: wp.array(dtype=int), hv: wp.vec3):
    """손(누르기): 잡은 입자의 옆 속도를 hv 로(기본 0 = 제자리), 위로는 못 뜨게. hv ≠ 0 = 손이 덩이를 끔(11-21 reports/60 확인 동작)."""
    i = wp.tid()
    p = idx[i]
    vel = v[p]
    v[p] = wp.vec3(hv[0], hv[1], wp.min(vel[2], 0.0))


@wp.kernel
def _pin(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), C: wp.array(dtype=wp.mat33),
         idx: wp.array(dtype=int), xp: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    p = idx[i]
    x[p] = xp[i]
    v[p] = wp.vec3(0.0, 0.0, 0.0)
    C[p] = wp.mat33(0.0)


@wp.kernel
def _soft_hold(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), idx: wp.array(dtype=int), xp: wp.array(dtype=wp.vec3),
               k: float, c: float, amax: float, dt: float, zclamp: int):
    """11-14 저녁 '부드러운 손': 잡은 입자를 잡을 때 자리로 끄는 용수철 + 감쇠, 입자마다 가속도 상한(= 손 힘 상한 / 잡은 질량).
    손이 버틸 수 있는 힘보다 세게 밀리면 그만큼 미끄러진다. press(속도만 0)는 위치가 새고, pin(위치 고정)은 고정 경계로 재료가 파고들었다(reports/41)."""
    i = wp.tid()
    p = idx[i]
    # 감쇠는 암시적(1 + c dt 로 나눔)이라 c 가 커도 안정 — c dt ≫ 1 이면 원래 손(press)처럼 거의 멈춰 버티고,
    # 용수철은 그동안 밀린 만큼을 시간 상수 c/k 로 천천히 되돌린다(11-14 저녁 2판).
    d = x[p] - xp[i]
    if zclamp == 1:  # 위아래는 원래 손처럼(위로 못 뜸, 아래로는 그대로), 용수철은 옆으로만(11-14 저녁 3판)
        d = wp.vec3(d[0], d[1], 0.0)
    vn = (v[p] - dt * k * d) / (1.0 + c * dt)
    if zclamp == 1:
        vn = wp.vec3(vn[0], vn[1], wp.min(v[p][2], 0.0))
    a = (vn - v[p]) / dt
    al = wp.length(a)
    if al > amax:
        a = a * (amax / al)
    v[p] = v[p] + dt * a


@wp.kernel
def _damp(v: wp.array(dtype=wp.vec3), C: wp.array(dtype=wp.mat33), flag: wp.array(dtype=int), s: float):
    p = wp.tid()
    if flag[p] == 1:
        v[p] = v[p] * s
        C[p] = C[p] * s


@wp.kernel
def _damp2(v: wp.array(dtype=wp.vec3), C: wp.array(dtype=wp.mat33), flag: wp.array(dtype=int), flag2: wp.array(dtype=int), s1: float, s2: float):
    p = wp.tid()
    s = float(1.0)
    if flag2[p] == 1:
        s = s2
    elif flag[p] == 1:
        s = s1
    if s < 1.0:
        v[p] = v[p] * s
        C[p] = C[p] * s


class Fixes:
    def __init__(self, sim, off, z_top, *, device, hold_box=None, hold_mode="press", sep_damp=300.0, sep_r=0.015,
                 sep_after=0.06, damp_from="cut_start", board_tol=0.6, step_dt=1e-3, adhesion=None, adh_release="top",
                 bounce_damp=0.0, bounce_h=1.5, release_outer=False, hold_k=1.0e4, hold_fmax=20.0, hold_zeta=1.0, hold_tau=0.0, hold_zclamp=False,
                 end_damp=0.0, end_h=0.008, end_scope="pair", spine_topple=0.0, spine_near=0.006, spine_cool=0.1, spine_max=3,
                 spine_min_n=50, spine_below=0.05):
        """off: 우리 좌표 → 솔버 좌표 이동. z_top: 재료 윗면(우리 좌표). hold_box: 손 상자 (cx, cy, cz, sx, sy, sz), 우리 좌표.
        board_tol: '도마에 닿음' 문턱, 입자 간격 배수(복사본 솔버일 때만)."""
        self.sim, self.off, self.z_top, self.device = sim, np.asarray(off, float), float(z_top), device
        self.hold_box, self.hold_mode = hold_box, hold_mode
        # hold_mode="soft"(11-14 저녁): 손 전체 강성 hold_k [N/m](잡은 입자에 고르게), 손 힘 상한 hold_fmax [N], 감쇠 비 hold_zeta(1 = 임계 감쇠).
        # 값은 가정: 손으로 채소를 도마에 누르고 잡을 때 10 N/mm·20 N 안팎(측정 없음).
        self.hold_k, self.hold_fmax, self.hold_zeta = float(hold_k), float(hold_fmax), float(hold_zeta)
        # hold_tau > 0(11-14 저녁 2판): 감쇠를 '밀린 만큼을 tau 초에 걸쳐 되돌리는' 세기로(c = k·tau). 꽉 잡고 버티다 천천히 제자리로.
        # 1판(임계 감쇠)은 칼을 들 때 몸통이 바로 되돌아가며 막 썬 3 mm 조각 아랫부분을 칼 바깥으로 밀어 떨어뜨렸다(tune/C1·C2).
        self.hold_tau = float(hold_tau)
        # hold_zclamp(11-14 저녁 3판): 위아래는 원래 손(press)처럼 위로 못 뜨게만 하고 용수철은 옆으로만. 1·2판은 칼을 들 때 몸통이 0.2 mm 같이 들리며
        # 막 썬 조각을 칼에서 떼어 냈다(tune/C1~C3).
        self.hold_zclamp = bool(hold_zclamp)
        self.sep_damp, self.sep_r, self.sep_after, self.damp_from, self.step_dt = sep_damp, sep_r, sep_after, damp_from, step_dt
        # end_damp [1/s] > 0(11-14 밤, 칼질 끝 감쇠): 날 끝이 도마 위 end_h 아래로 내려온 뒤부터 dwell 끝까지 칼 옆 감쇠 세기를 sep_damp 대신 이 값으로.
        # 팀원 솔버는 칼(두껍게 친 날 끝)이 격자점 한 줄을 눌러 쌓은 탄성 에너지를 균열이 그 줄을 지날 때 한꺼번에 푼다(힘 0 → 40 N → 0 톱니, 간격 = 1칸).
        # 그 에너지는 이미 균열 값(G_c × 넓이)으로 치렀는데 재료에 그대로 남아 있다가(이중 계산) 마지막 줄이 풀릴 때 막 떨어진 조각을 튕겨 낸다
        # (당근 3 mm 조각이 옆 2.8 mm·위 2.4 mm, reports/45). 그 순간만 세게 감쇠해 그 에너지를 열로 뺀다(규칙).
        # end_scope="pair": 세게 감쇠하는 것은 지금 칼질이 가르는 짝(몸통과 막 썬 조각)만 — 칼에 붙어 같이 내려오는 앞 조각까지 멈추면 칼 위에서 미끄러진다.
        # "all": 칼 옆 감쇠 대상 전부(첫 시험 E1).
        self.end_damp, self.end_h, self.end_scope = float(end_damp), float(end_h), str(end_scope)
        self.damp_rate = float(sep_damp)
        self.end_events = []
        self.end_on, self.end_ids, self.end_flag = False, set(), None
        self.spacing = sim.dx / sim.config.particles_per_axis
        self.hold = {"on": False}
        self.hold_vel = (0.0, 0.0, 0.0)  # 11-21 reports/60: 손(press)이 잡은 입자를 이 옆 속도 [m/s]로 끔('pull' 확인 동작), 0 = 원래 누르기
        self.damp = {"on": False, "flag": None}
        self.state = {"damping": False, "until": -1, "events": [], "steps": 0}
        self.adhesion = dict(adhesion) if adhesion else None
        if self.adhesion is not None:
            sp = sim.dx / sim.config.particles_per_axis
            self.adhesion.setdefault("d_c", 0.5 * sp)  # 입자 반지름: 이보다 가까우면 '닿음'
            if self.adhesion.get("layers"):  # 11-13: 물막은 맞닿은 면에만 — 접착 구역 = 칼 면 접촉 거리 + layers 입자 간격(면적에 비례하게)
                self.adhesion["r0"] = self.adhesion["d_c"]
                self.adhesion["r1"] = self.adhesion["d_c"] + float(self.adhesion["layers"]) * sp
            # 11-17 조각 면 맞춤(anchor, reports/47): 기본 꺼짐. 켜면 조각·쪽마다 칼 쪽 면(입자 거리 anchor_q 백분위)을 재서 구역을 그 면에 맞춘다.
            # anchor_lo~anchor_hi [m] 밖으로는 안 옮김(그보다 멀면 떨어지는 중 — 구역이 그대로 칼 면에 남아 접착이 약해짐).
            self.adhesion.setdefault("anchor", 0)
            self.adhesion.setdefault("anchor_q", 0.1)
            self.adhesion.setdefault("anchor_lo", -0.3e-3)
            self.adhesion.setdefault("anchor_hi", 0.6e-3)
        self.adh_anchor = int(self.adhesion["anchor"]) if self.adhesion is not None else 0
        self.adh_nb, self.adh_bw, self.adh_dlo = 200, 0.02e-3, -1.0e-3  # 막대그림: −1.0 ~ +3.0 mm, 0.02 mm 칸
        n_side = 2 * int(sim.config.max_pieces)
        self.adh_shift = wp.zeros(n_side, dtype=float, device=device)
        self.adh_hist = wp.zeros(n_side * self.adh_nb, dtype=int, device=device) if self.adh_anchor else None
        # adh_release="top": 칼이 맨 위에서 멈추는 단계(top·move·rest)에는 접착을 끈다 → 붙어 올라간 조각이 거기서 떨어진다(규칙, 11-13).
        # 끄지 않으면("none") 조각이 칼에 실려 다음 칼질 자리로 가서 오이 윗면에 걸려 칼등(35 mm) 위로 넘어가 칼 양쪽에 걸치고,
        # 팀원 솔버 칼 접촉은 한쪽에만 있는 조각을 전제로 해 칼을 들 때 칼날이 그 조각을 뚫는다(tune/cucumber_film_split2, 최대 226 입자).
        self.adh_release = adh_release
        self.adh_on = True
        # bounce_damp [1/s] > 0: 칼에 안 붙은 조각이 도마에 닿아 있는 동안(입자 하나라도 도마 면 bounce_h·입자 간격 안) 그 조각 전체의
        # 위쪽 속도를 단계마다 1/(1 + c dt) 배(규칙, 11-13).
        # 팀원 오이 재료는 거의 순수 탄성(점성 2 Pa·s)이라 칼에서 54 mm 떨어진 조각이 거의 같은 높이로 튀어 올랐다(반발 계수 약 0.9).
        self.bounce_damp, self.bounce_h = float(bounce_damp), float(bounce_h)
        # release_outer(11-14 오후): 칼을 드는 동안(withdraw) 칼 바깥쪽(+법선 쪽 = 잘려 나간 쪽) 조각은 칼 옆 감쇠에서 뺀다.
        # 감쇠(시간 상수 3.3 ms)가 칼에 붙어 올라가야 할 작은 조각(채)의 속도까지 지워, 날 끝이 지나가기 전에 못 따라오게 했다(reports/39).
        self.release_outer = bool(release_outer)
        self._phase = ""
        self.piece_flag = wp.zeros(int(sim.config.max_pieces), dtype=int, device=device)
        self.piece_touch = wp.zeros(int(sim.config.max_pieces), dtype=int, device=device)
        # spine_topple [rad/s] > 0(11-19 저녁, reports/55): 칼등 위 넘어뜨리기(규칙).
        # 칼에 붙은 편이 다음 편에 밀려 칼 면을 따라 올라가면, 칼등 위에서는 편끼리 가장자리를 딛고 1~2초 세로로 섰다(reports/54 '칼등 위 기둥').
        # 칼등 위엔 접착도 칼 접촉도 없어 받침은 밑의 편뿐이고, 계산이 좌우 대칭이라 넘어질 계기가 없다(실물은 세운 동전처럼 곧 넘어짐 — 가정).
        # 그래서 1 ms 단계마다 조각이 거의 다(칼등보다 낮은 입자 spine_below 이하) 칼등 위에 있고 무게중심이 칼 평면에서 spine_near 안이면,
        # 그 조각의 가장 낮은 점을 축으로 칼에서 먼 쪽(무게중심이 있는 쪽)으로 넘어지게 각속도 spine_topple 을 한 번 더한다(대칭 깨기).
        # 다시 걸리면 spine_cool 초 뒤 또, 조각마다 spine_max 번까지.
        # 첫 판(T3, 무게중심이 칼등보다 높으면 걸기)은 칼에 붙은 둘째 편의 무게중심이 칼등 높이에 딱 걸쳐(아래 절반은 칼 면에 붙음) 그때 3번을 다 써
        # 정작 완전히 올라갔을 때 못 걸었다(reports/55 2절).
        self.spine_topple, self.spine_near, self.spine_below = float(spine_topple), float(spine_near), float(spine_below)
        self.spine_cool_steps, self.spine_max, self.spine_min_n = int(round(spine_cool / step_dt)), int(spine_max), int(spine_min_n)
        self.spine_events = []
        self.sp_free = []
        self.sp_released = set()  # 4판(T3d): 한 번 칼등 위로 다 올라갔던 조각은 끝까지 칼 옆 감쇠에서 뺌
        if self.spine_topple > 0:
            npc = int(sim.config.max_pieces)
            self.sp_cnt = wp.zeros(npc, dtype=float, device=device)
            self.sp_suvw = wp.zeros(npc, dtype=wp.vec3, device=device)
            self.sp_wmin = wp.zeros(npc, dtype=float, device=device)
            self.sp_below = wp.zeros(npc, dtype=float, device=device)
            self.sp_om = wp.zeros(npc, dtype=wp.vec3, device=device)
            self.sp_pv = wp.zeros(npc, dtype=wp.vec3, device=device)
            self.sp_n = np.zeros(npc, int)
            self.sp_last = np.full(npc, -10 ** 9)
        self.adh_stats = wp.zeros(2, dtype=float, device=device)
        self.board_contact = hasattr(sim, "board_minz")
        if self.board_contact:
            sim.board_tol = float(board_tol * self.spacing)
        orig = sim.substep

        def _wl(kernel, dim, inputs, device=None):
            wp.launch(kernel, dim=dim, inputs=inputs, device=device)

        def substep(dt):
            orig(dt)
            L = sim._L if getattr(sim, "fast_launch", 0) else _wl  # 11-23 reports/62: 빠른 판은 미리 만든 Launch 다시 쓰기
            if self.damp["on"]:
                if self.end_on and self.end_flag is not None:  # 짝만 세게, 나머지는 원래 칼 옆 감쇠
                    L(_damp2, dim=sim.n_particles, inputs=[sim.v, sim.C, self.damp["flag"], self.end_flag,
                              1.0 / (1.0 + self.sep_damp * dt), 1.0 / (1.0 + self.end_damp * dt)], device=device)
                else:
                    L(_damp, dim=sim.n_particles, inputs=[sim.v, sim.C, self.damp["flag"], 1.0 / (1.0 + self.damp_rate * dt)],
                              device=device)
            if self.adhesion is not None and self.adh_on:
                a = self.adhesion
                if self.adh_anchor:
                    self.adh_hist.zero_()
                    L(_adh_gap_hist, dim=sim.n_particles, inputs=[sim.x, sim.piece, sim._knife, a["r1"] + a["anchor_hi"], a["min_kw"],
                              self.adh_dlo, 1.0 / self.adh_bw, self.adh_nb, self.adh_hist], device=device)
                    L(_adh_gap_pct, dim=self.adh_shift.shape[0], inputs=[self.adh_hist, self.adh_nb, float(a["anchor_q"]), self.adh_dlo,
                              self.adh_bw, float(a["anchor_lo"]), float(a["anchor_hi"]), 20, self.adh_shift], device=device)
                L(_adhere, dim=sim.n_particles, inputs=[sim.x, sim.v, sim._knife, dt, float(sim.material.density), a["r0"],
                          a["r1"], a["sigma"], a["tau"], a["c_n"], a["c_t"], a["p_a"], a["min_kw"], a["d_c"], self.adh_stats,
                          sim.piece, self.adh_shift, self.adh_anchor],
                          device=device)
            if self.bounce_damp > 0:
                self.piece_flag.zero_()
                if self.adhesion is not None and self.adh_on:
                    a = self.adhesion
                    L(_adh_flag, dim=sim.n_particles, inputs=[sim.x, sim.piece, sim._knife, a["r1"], a["min_kw"], self.piece_flag],
                              device=device)
                self.piece_touch.zero_()
                L(_board_flag, dim=sim.n_particles, inputs=[sim.x, sim.piece, float(sim.board_z), self.bounce_h * self.spacing,
                          self.piece_touch], device=device)
                L(_bounce, dim=sim.n_particles, inputs=[sim.v, sim.piece, self.piece_flag, self.piece_touch,
                          1.0 / (1.0 + self.bounce_damp * dt)], device=device)
            if self.hold["on"]:
                if self.hold_mode == "press":
                    L(_press, dim=self.hold["n"], inputs=[sim.v, self.hold["idx"], wp.vec3(*self.hold_vel)], device=device)
                elif self.hold_mode == "soft":
                    h = self.hold
                    L(_soft_hold, dim=h["n"], inputs=[sim.x, sim.v, h["idx"], h["xp"], h["k"], h["c"], h["amax"], dt,
                                                              1 if self.hold_zclamp else 0],
                              device=device)
                else:
                    L(_pin, dim=self.hold["n"], inputs=[sim.x, sim.v, sim.C, self.hold["idx"], self.hold["xp"]],
                              device=device)

        sim.substep = substep

    def grip(self):
        """자리 잡은 뒤 한 번: 손 상자 x-y 넓이 안 입자를 손으로 잡는다. returns 잡은 입자 수."""
        if self.hold_box is None:
            return 0
        cx, cy, cz, sx, sy, sz = self.hold_box
        xo = self.sim.positions() - self.off
        sel = np.flatnonzero((np.abs(xo[:, 0] - cx) <= sx / 2) & (np.abs(xo[:, 1] - cy) <= sy / 2))
        self.hold.update(on=True, n=len(sel), idx=wp.array(sel.astype(np.int32), dtype=int, device=self.device),
                         xp=wp.array((xo[sel] + self.off).astype(np.float32), dtype=wp.vec3, device=self.device))
        if self.hold_mode == "soft" and len(sel):  # 손 전체 강성·힘 상한을 잡은 질량으로 나눠 입자마다(11-14 저녁)
            M = len(sel) * float(self.sim.particle_mass)
            c = (self.hold_k / M) * self.hold_tau if self.hold_tau > 0 else 2.0 * self.hold_zeta * math.sqrt(self.hold_k / M)
            self.hold.update(k=self.hold_k / M, c=c, amax=self.hold_fmax / M, mass=M)
        return len(sel)

    def _select_fast(self, q, cKDTree):
        """_select 와 같은 답(11-23 reports/62, tools/check_select.py 로 reports/61 열한 판 프레임에서 확인).
        - 조각 번호는 30 미만 정수라 np.unique 대신 개수 세기(bincount), 조각별 입자는 16비트 안정 정렬(기수 정렬) 한 번으로 묶음
          (안정 정렬이라 조각 안 순서 = 원래 순서 → 나무·중앙값 입력이 원래와 같음).
        - 맞닿은 조각 따라가기: 두 조각을 감싼 상자가 어느 축으로든 touch 보다 멀면 어떤 두 입자도 touch 안에 못 들므로 거리 계산을 건너뛰고,
          나무(cKDTree)는 실제로 물을 때만 만든다(옛 판은 매 단계 모든 조각 나무를 만듦, 22 ms)."""
        pos, piece = self.sim.positions(), self.sim.pieces()
        nrm = np.array([math.cos(q[3]), math.sin(q[3]), 0.0])
        sd = (pos - (np.asarray(q[:3]) + self.off)) @ nrm
        near = np.abs(sd) < self.sep_r
        cnt = np.bincount(piece)
        ids = np.flatnonzero(cnt)
        out = set(np.flatnonzero(np.bincount(piece[near], minlength=len(cnt))).tolist())
        todo = list(out)
        touch = 1.5 * self.spacing
        order = np.argsort(piece.astype(np.int16), kind="stable")
        start = np.concatenate([[0], np.cumsum(cnt)])
        grp = {int(k): order[start[k]:start[k + 1]] for k in ids}
        lo = {k: pos[v].min(0) for k, v in grp.items()}
        hi = {k: pos[v].max(0) for k, v in grp.items()}
        lim = touch * (1.0 + 1e-9)
        trees = {}
        while todo:
            a = int(todo.pop())
            for b in ids:
                b = int(b)
                if b in out:
                    continue
                if max(float((lo[b] - hi[a]).max()), float((lo[a] - hi[b]).max())) > lim:
                    continue
                if a not in trees:
                    trees[a] = cKDTree(pos[grp[a]])
                if trees[a].query(pos[grp[b]], distance_upper_bound=touch)[0].min() < touch:
                    out.add(b)
                    todo.append(b)
        if self.release_outer and self._phase.startswith("withdraw"):
            out -= {k for k in grp if np.median(sd[grp[k]]) > 0.0}
        lut = np.zeros(len(cnt), dtype=bool)
        lut[list(out)] = True
        return lut[piece]

    def _select_fast2(self, q, cKDTree):
        """_select_fast 와 같은 답(11-23 reports/63 tools/check_select2.py). 빠른 판에서도 시간의 70~76 % 가 조각마다
        pos[입자 번호].min/max(조각 수만큼 입자 모으기)였다 → 정렬한 자리를 한 번 모아 reduceat 으로 조각별 최소·최대
        (최소·최대는 순서와 무관하게 같은 값). 나머지(맞닿은 조각 따라가기·나무·중앙값)는 _select_fast 그대로."""
        pos, piece = self.sim.positions(), self.sim.pieces()
        nrm = np.array([math.cos(q[3]), math.sin(q[3]), 0.0])
        sd = (pos - (np.asarray(q[:3]) + self.off)) @ nrm
        near = np.abs(sd) < self.sep_r
        cnt = np.bincount(piece)
        ids = np.flatnonzero(cnt)
        out = set(np.flatnonzero(np.bincount(piece[near], minlength=len(cnt))).tolist())
        todo = list(out)
        touch = 1.5 * self.spacing
        order = np.argsort(piece.astype(np.int16), kind="stable")
        start = np.concatenate([[0], np.cumsum(cnt)])
        grp = {int(k): order[start[k]:start[k + 1]] for k in ids}
        ps = pos[order]
        st = start[ids]
        lo_a = np.minimum.reduceat(ps, st, axis=0)
        hi_a = np.maximum.reduceat(ps, st, axis=0)
        lo = {int(k): lo_a[n] for n, k in enumerate(ids)}
        hi = {int(k): hi_a[n] for n, k in enumerate(ids)}
        lim = touch * (1.0 + 1e-9)
        trees = {}
        while todo:
            a = int(todo.pop())
            for b in ids:
                b = int(b)
                if b in out:
                    continue
                if max(float((lo[b] - hi[a]).max()), float((lo[a] - hi[b]).max())) > lim:
                    continue
                if a not in trees:
                    trees[a] = cKDTree(pos[grp[a]])
                if trees[a].query(pos[grp[b]], distance_upper_bound=touch)[0].min() < touch:
                    out.add(b)
                    todo.append(b)
        if self.release_outer and self._phase.startswith("withdraw"):
            out -= {k for k in grp if np.median(sd[grp[k]]) > 0.0}
        lut = np.zeros(len(cnt), dtype=bool)
        lut[list(out)] = True
        return lut[piece]

    def _select(self, q):
        from scipy.spatial import cKDTree

        fs = getattr(self, "fast_select", False)
        if fs and int(fs) >= 2:  # 11-23 reports/63: 조각 상자를 reduceat 으로(같은 답, tools/check_select2.py)
            return self._select_fast2(q, cKDTree)
        if fs:  # 11-23 reports/62: 같은 답(tools/check_select.py), 빠른 판
            return self._select_fast(q, cKDTree)
        pos, piece = self.sim.positions(), self.sim.pieces()
        nrm = np.array([math.cos(q[3]), math.sin(q[3]), 0.0])
        sd = (pos - (np.asarray(q[:3]) + self.off)) @ nrm
        near = np.abs(sd) < self.sep_r
        ids = np.unique(piece)
        out = set(np.unique(piece[near]).tolist())
        todo = list(out)
        touch = 1.5 * self.spacing
        trees = {k: cKDTree(pos[piece == k]) for k in ids}
        while todo:
            a = todo.pop()
            for b in ids:
                if b not in out and trees[a].query(pos[piece == b], distance_upper_bound=touch)[0].min() < touch:
                    out.add(b)
                    todo.append(b)
        if self.release_outer and self._phase.startswith("withdraw"):
            out -= {int(k) for k in ids if np.median(sd[piece == k]) > 0.0}
        return np.isin(piece, list(out))

    def update(self, i, phase, end_of_phase, q):
        """1 ms 단계 i 를 진행한 뒤 부른다. phase = 이 단계 이름(우리 T4 단계), q = 칼 날 끝 중심 x·y·z·회전(우리 좌표)."""
        self._phase = str(phase)
        if self.spine_topple > 0:
            self._spine_topple(i, phase)
        if self.adh_release == "top":
            self.adh_on = not str(phase).startswith(("top", "move", "rest"))
        if self.sep_damp <= 0:
            return
        if self.end_damp > 0:  # 칼질 끝 감쇠 구간: cut 단계에서 날 끝이 end_h 아래, 그리고 dwell 단계
            ph_ = str(phase)
            end_on = (ph_.startswith("cut") and q[2] < self.end_h) or ph_.startswith(("dwell", "bsaw"))  # 11-21: 바닥 톱질도
            if self.end_scope == "all":
                rate = self.end_damp if end_on else self.sep_damp
                if rate != self.damp_rate:
                    self.end_events.append({"step": int(i), "phase": ph_, "rate": rate})
                self.damp_rate = rate
            else:
                sim_ = self.sim
                if end_on and not self.end_on:  # 구간 시작: 지금 칼질의 짝 조각 번호를 적어 둠(칼질이 끝나면 짝 목록에서 빠지므로)
                    c_ = sim_.active_cut
                    self.end_ids = {int(a) for a, b, cc in sim_._pairs if cc == c_} | {int(b) for a, b, cc in sim_._pairs if cc == c_}
                    self.end_events.append({"step": int(i), "phase": ph_, "pieces": sorted(self.end_ids)})
                if end_on and self.end_ids:
                    sel_ = np.isin(sim_.pieces(), list(self.end_ids))
                    self.end_flag = wp.array(sel_.astype(np.int32), dtype=int, device=self.device)
                elif not end_on and self.end_on:
                    self.end_events.append({"step": int(i), "phase": ph_, "off": True})
                self.end_on = bool(end_on and self.end_ids)
        st, sim = self.state, self.sim
        if str(phase).startswith("cut") and not st["damping"]:
            if self.damp_from == "cut_start" or end_of_phase or (sim.active_cut >= 0 and sim.cut_complete(sim.active_cut)):
                st["damping"] = True
                st["events"].append({"step": int(i), "phase": str(phase)})
        if st["damping"] and q[2] > self.z_top:
            st["damping"], st["until"] = False, i + int(round(self.sep_after / self.step_dt))
        if st["damping"] or i <= st["until"]:
            sel = self._select(q)
            if self.spine_topple > 0 and self.sp_released:  # 칼등 위로 다 올라갔던 조각은 감쇠 안 함(_spine_topple 3·4판)
                sel &= ~np.isin(self.sim.pieces(), sorted(self.sp_released))
            self.damp.update(on=True, flag=wp.array(sel.astype(np.int32), dtype=int, device=self.device))
            st["steps"] += 1
        elif self.damp["on"]:
            self.damp["on"] = False

    def _spine_topple(self, i, phase):
        """칼등 위 넘어뜨리기(규칙, __init__ 의 spine_topple 설명)."""
        sim, k = self.sim, self.sim._knife
        self.sp_free = []
        if int(k.enabled) != 1:
            return
        self.sp_cnt.zero_()
        self.sp_suvw.zero_()
        self.sp_below.zero_()
        self.sp_wmin.fill_(1.0e9)
        wp.launch(_piece_knife_stats, dim=sim.n_particles, inputs=[sim.x, sim.piece, k, self.sp_cnt, self.sp_suvw, self.sp_wmin,
                                                                     self.sp_below], device=self.device)
        cnt, suvw, wmin, nbelow = self.sp_cnt.numpy(), self.sp_suvw.numpy(), self.sp_wmin.numpy(), self.sp_below.numpy()
        hl, H, taper, tip = float(k.half_length), float(k.height), float(k.spine_taper), float(k.tip_rise)
        kick = []
        for f in np.flatnonzero(cnt >= self.spine_min_n):
            u, v, w = (suvw[f] / cnt[f]).tolist()
            below = float(nbelow[f] / cnt[f])
            if abs(u) > hl or abs(v) > self.spine_near or below > self.spine_below:
                continue
            # 3판(T3c): 칼등 위로 다 올라간 조각은 칼 옆 감쇠(시간 상수 3.3 ms)에서 뺀다 — update() 가 감쇠 대상에서 지움.
            # 2판(T3b)은 칼질 중 감쇠가 넘어뜨리는 속도를 몇 ms 안에 지워(윗끝 0.1초에 0.2~0.5 mm), 칼을 들 때 감쇠가 풀려야 넘어졌다.
            # 4판(T3d): 3판은 기울어 무게중심이 칼 평면 6 mm 밖으로 나가는 순간 다시 감쇠돼 14° 에서 0.4초 버텼다 → 한 번 빼면 끝까지 뺌(sp_released).
            self.sp_free.append(int(f))
            self.sp_released.add(int(f))
            if self.sp_n[f] >= self.spine_max or i - self.sp_last[f] < self.spine_cool_steps:
                continue
            if taper > 0:  # 기록용: 무게중심 자리의 칼등 높이(mpm._spine_w 와 같은 모양)
                s_ = min(max((u - (hl - taper)) / taper, 0.0), 1.0)
                spine = tip + (H - tip) * (1.0 - s_ * s_)
            else:
                spine = H
            kick.append((int(f), u, v, w, spine, below))
        if not kick:
            return
        o = np.array(k.origin, float)
        eu, ev, ew = np.array(k.e_u, float), np.array(k.e_v, float), np.array(k.e_w, float)
        axis = np.cross(ew, ev)
        axis /= np.linalg.norm(axis)  # axis × e_w = e_v: 아래 끝을 축으로 돌리면 윗부분이 +e_v 로 감
        npc = len(cnt)
        om, pv = np.zeros((npc, 3), np.float32), np.zeros((npc, 3), np.float32)
        for f, u, v, w, spine, below in kick:
            sgn = 1.0 if v >= 0.0 else -1.0
            om[f] = sgn * self.spine_topple * axis
            pv[f] = o + u * eu + v * ev + float(wmin[f]) * ew
            self.sp_n[f] += 1
            self.sp_last[f] = i
            self.spine_events.append({"step": int(i), "phase": str(phase), "piece": f, "n": int(cnt[f]), "com_w_mm": round(w * 1e3, 1),
                                      "spine_mm": round(spine * 1e3, 1), "com_v_mm": round(v * 1e3, 2), "low_w_mm": round(float(wmin[f]) * 1e3, 1),
                                      "below": round(below, 3)})
        self.sp_om.assign(om)
        self.sp_pv.assign(pv)
        wp.launch(_topple_kick, dim=sim.n_particles, inputs=[sim.x, sim.v, sim.C, sim.piece, self.sp_om, self.sp_pv], device=self.device)

    def adhesion_read(self):
        """지난 읽기(= 1 ms 단계 하나) 뒤 쌓인 [접착 걸린 입자 수(작은 단계 합), 입자에 건 위쪽 힘 N(그 1 ms 평균)] 를 읽고 비운다.
        11-13 정정: 처음엔 힘을 작은 단계 dt(≈2e-5 s)로 나눠 약 48배 크게 적었다(tune 첫 다섯 판의 adh_og[:,1]). 지금은 step_dt 로 나눈다."""
        a = self.adh_stats.numpy().copy()
        self.adh_stats.zero_()
        return a[0], a[1] * self.sim.particle_mass / max(self.step_dt, 1e-12)

    def summary(self):
        return {"adhesion": self.adhesion, "adh_release": self.adh_release, "release_outer": self.release_outer, "bounce_damp": [self.bounce_damp, self.bounce_h], "hold_mode": self.hold_mode, "hold_soft": ([self.hold_k, self.hold_fmax, self.hold_zeta, self.hold_tau, self.hold_zclamp] if self.hold_mode == "soft" else None), "hold_box": self.hold_box, "held": int(self.hold.get("n", 0)),
                "sep_damp": [self.sep_damp, self.sep_r, self.sep_after, self.damp_from], "damp_events": self.state["events"],
                "end_damp": [self.end_damp, self.end_h, self.end_scope], "end_events": self.end_events[:40],
                "damped_steps": self.state["steps"], "board_contact_tol_m": float(self.sim.board_tol) if self.board_contact else None,
                "spine_topple": ([self.spine_topple, self.spine_near, self.spine_cool_steps, self.spine_max, self.spine_below]
                                 if self.spine_topple > 0 else None),
                "spine_events": self.spine_events[:60], "spine_released": sorted(self.sp_released)}
