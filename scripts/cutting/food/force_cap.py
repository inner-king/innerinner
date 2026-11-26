"""칼 힘 한계(어드미턴스) — 칼이 받은 저항이 한계에 다가가면 그 1 ms 동안 덜 내려가고, 한계를 넘으면 멈춘다 (2026-11-24, reports/65).

계획: docs/plan_2026-11-24_force_pipeline_grasp.md 3절. 우리 실행기(og_veg_cut.py)와 팀원 파이프라인 새 실행기(단계 B)가 같이 쓴다.

무엇
- 로봇 팔이 낼 수 있는 힘이 모자라면 칼이 재료에 안 들어가고 멈춘다. 지금까지 시뮬 칼은 정해진 경로대로만 움직여(힘은 기록만) 어떤 재료든 늘 썰렸다.
- 이 모듈은 '이번 1 ms 에 계획된 내려가기의 몇 할을 허락할지'(rate, 0~1)를 칼이 받은 위쪽 힘으로 정한다.
  - admit(기본): rate = (한계 − 힘) / (band × 한계) 를 0~1 로 자름. 힘이 한계의 (1 − band) 아래면 계획대로, 한계에 가까울수록 천천히, 한계 이상이면 0.
  - hold: 한계 아래면 1, 이상이면 0(켜고 끄기).
- **admit 을 기본으로 한 까닭(11-24)**: 팀원 솔버의 밀며 썰기 식(mpm.py `_slice_factor`)은 '칼이 멈춰 있으면 아무것도 안 잘림'이고,
  잘릴 때 필요한 에너지는 1/(1+ξ²)배(ξ = 칼날 방향 빠르기 ÷ 누르는 빠르기, 최대 SimConfig.slice_push_max = 2)다.
  hold 는 내려갈 때마다 계획 빠르기(40 mm/s)로 눌러 ξ 가 작게 잡힌다 → 밀며 썰기 이득을 덜 본다. admit 은 천천히 눌러 실제 힘 제한 팔처럼 ξ 가 커진다.
- 멈춤 판정: 칼질 하나에서 stall_s 초 동안 칼 높이가 stall_dz 만큼도 안 내려가거나, 덜 내려간 시간(계획보다 늦어진 시간)이 stall_max_s 를 넘으면 '멈춤'.
  멈춘 뒤 처리(남은 경로를 멈춘 높이 아래로 안 내려가게 하는 것 등)는 부르는 쪽이 한다.

ElRobot 값(reports/28 7행): 아래로 누르는 힘 콘솔 제한 12.1 N, 모터 중앙값 18.6 N — 칼을 쥐지 않은 정적 기록값(계산값 아님).
실물로 다시 재면 PRESETS 만 고친다.
"""
from collections import deque

PRESETS = {
    "elrobot_console": 12.1,  # ElRobot 콘솔 제한(reports/28, 09 실측 정적 기록값)
    "elrobot_motor": 18.6,    # ElRobot 모터 중앙값(같음)
}


def parse_cap(v):
    """'none'·'off'·'0'·None → None, 프리셋 이름 → 그 값, 그 밖 → float(N)."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("", "none", "off", "0", "0.0"):
        return None
    if s in PRESETS:
        return float(PRESETS[s])
    try:
        x = float(s)
    except ValueError as e:
        raise ValueError(f"힘 한계 '{v}': 숫자(N) 또는 {sorted(PRESETS)} 또는 none") from e
    if x <= 0:
        return None
    return x


class ForceGate:
    """칼 위쪽 힘(Fz, 재료가 칼을 밀어 올리는 힘, +)을 받아 내려가기 몫을 정하고 멈춤을 판정한다.

    부르는 순서(1 ms 단계마다): rate() 로 이번 단계 몫을 받고 → 칼을 움직여 솔버를 한 단계 → push_force(fz) → (칼질 중이면) note(z, t).
    칼질 하나를 시작할 때 start_cut(k, z, t).
    """

    def __init__(self, cap_N, *, mode="admit", band=0.2, filter_steps=5, stall_s=2.0, stall_dz=2.0e-4, stall_max_s=10.0, step_dt=1.0e-3):
        if cap_N is None or cap_N <= 0:
            raise ValueError("ForceGate 는 양수 한계가 필요(한계 없음이면 만들지 말 것)")
        if mode not in ("admit", "hold"):
            raise ValueError(f"mode '{mode}': admit 또는 hold")
        self.cap = float(cap_N)
        self.mode = mode
        self.band = max(float(band), 1.0e-6)
        self.filter_steps = max(int(filter_steps), 1)
        self.stall_s = float(stall_s)
        self.stall_dz = float(stall_dz)
        self.stall_max_s = float(stall_max_s)
        self.step_dt = float(step_dt)
        self._buf = deque(maxlen=self.filter_steps)
        self._sum = 0.0
        self.k = -1
        self.z_ref = None
        self.t_ref = 0
        self.slow_steps = 0  # 이번 칼질에서 계획보다 덜 내려간 단계 수

    # 힘 거르기: 최근 filter_steps 단계 평균(솔버 힘에는 칼날 끝이 새 격자점 줄에 닿을 때마다 생기는 톱니가 있어 그대로 쓰면 떨림)
    def push_force(self, fz):
        fz = float(fz)
        if len(self._buf) == self._buf.maxlen:
            self._sum -= self._buf[0]
        self._buf.append(fz)
        self._sum += fz

    @property
    def f_filt(self):
        return self._sum / len(self._buf) if self._buf else 0.0

    def rate(self):
        f = self.f_filt
        if self.mode == "hold":
            return 1.0 if f < self.cap else 0.0
        r = (self.cap - f) / (self.band * self.cap)
        return 1.0 if r >= 1.0 else (0.0 if r <= 0.0 else r)

    def start_cut(self, k, z, t):
        self.k = int(k)
        self.z_ref = float(z)
        self.t_ref = int(t)
        self.slow_steps = 0

    def note(self, z, t, slow):
        """칼질 중 한 단계 뒤: 칼 높이 z [m], 솔버 시간 t [단계], 이번 단계가 계획보다 덜 내려갔나(slow). 멈춤이면 (True, 까닭)."""
        if slow:
            self.slow_steps += 1
        if self.z_ref is None or z < self.z_ref - self.stall_dz:
            self.z_ref, self.t_ref = float(z), int(t)
        if (t - self.t_ref) * self.step_dt >= self.stall_s:
            return True, f"{self.stall_s:g} s 동안 {self.stall_dz * 1e3:g} mm 도 안 내려감"
        if self.slow_steps * self.step_dt >= self.stall_max_s:
            return True, f"늦어진 시간 {self.stall_max_s:g} s 넘음"
        return False, ""

    def summary(self):
        return {"cap_N": self.cap, "mode": self.mode, "band": self.band, "filter_steps": self.filter_steps, "stall_s": self.stall_s,
                "stall_dz_m": self.stall_dz, "stall_max_s": self.stall_max_s}
