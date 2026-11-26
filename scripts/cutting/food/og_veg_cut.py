"""OmniGibson 장면 안에서 팀원 Warp MPM 솔버로 채소를 써는 영상 (2026-11-13, reports/35).

  OMNIGIBSON_HEADLESS=1 OMNIGIBSON_DATA_PATH=$HOME/innerinner/deps/BEHAVIOR-1K/datasets \\
  PYTHONPATH=$PWD python scripts/cutting/food/og_veg_cut.py \\
      --veg cucumber --out reports/35_og_veg_cut/cucumber

11-24 사용자 결정: 결과 같은 속도 고침 '빠른3'(reports/62·63, 오이 영상 3.4배·식재료 열 가지 3.1~5.0배)이 기본이다.
  PYTHONPATH 의 원본 솔버(<빠른3 솔버 복사본>·7·8)를 짝 복사본 <원본>_fast3 로 저절로 바꿔 쓴다(patches/build_fast3_overlay.sh 가 만듦·도장 확인).
  원래 코드로 돌리려면 --no_fast3. 원본 솔버를 고치면 빠른 복사본을 다시 만들 것(안 하면 경고 뒤 원래 속도로 돎).

무엇
- OmniGibson(Isaac Sim) 빈 장면에 탁자·도마·칼을 놓고, 같은 프로세스에서 팀원 솔버(`simfoundry/cutting`, 도마 접촉 고친 복사본)를 돌린다.
  채소는 팀원 `og_cosim.FoodPoints`(입자마다 작은 구)로 보인다.
- 고친 판(reports/34 7 절 최종): 손 '누르기'(가까이) + 칼 옆 조각 감쇠(칼질 처음부터) + 도마 '닿은 재료만 받침'(문턱 0.6 입자 간격).
  `--no_fixes` 면 팀원 솔버 그대로(비교용, 이때는 원본 솔버를 PYTHONPATH 에 둘 것).
- 칼: 로봇 없이 정해진 경로(우리 Genesis 실행의 칼 기록 log.npz q, 1 ms 마다)를 그대로 따른다. OmniGibson 의 칼 물체는 그 자세로 옮겨 그리기만 한다
  (visual_only — 다음 단계에서 Franka 가 쥐게 할 때 PhysX 강체로 바꾼다). 칼 모양은 우리 칼(두께 1.5 mm, 날 끝 0.2 mm, 높이 35 mm, 길이 80 mm).
- 기록: 카메라 3대 영상(`<cam>.mp4`), 우리 진단(`scripts/veg/diag_*.py`·`scripts/cutting/food/summarize_runs.py`)이 읽는 frames.npz·log.npz, metrics.json.
"""
import argparse
import json
import math
import os

# 11-23 reports/62: numpy(OpenBLAS)가 매 단계 작은 행렬 곱 때문에 일꾼 스레드 15개를 늘 바쁘게 기다리게 해(프로세스 CPU 906 %) 주 스레드를
# 방해했다 → 1개로(결과 불변, 오이 앞 4칼질 1.24배 빠름). numpy 를 부르기 전에 정해야 함. 바꾸려면 환경 변수를 직접 줄 것.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

VEG = {  # 우리 data/veg/meshes 와 같은 크기. 손 상자 = scripts/veg/run_adhesion.sh 의 near(마지막 칼질 8~9 mm 뒤까지)
    "cucumber": dict(L=0.070, r0=0.015, r1=0.015, material="cucumber", hold=(-0.0135, 0, 0.040, 0.043, 0.040, 0.020),
                     genesis="reports/33_veg_cut_adhesion/adhesion/T4_cucumber_off", skin=(0.10, 0.24, 0.08),
                     flesh_in=(0.92, 0.92, 0.80), flesh_out=(0.80, 0.86, 0.55)),
    "zucchini": dict(L=0.072, r0=0.018, r1=0.018, material="zucchini", hold=(-0.0135, 0, 0.046, 0.045, 0.045, 0.020),
                     genesis="reports/33_veg_cut_adhesion/adhesion/T4_zucchini_strong", skin=(0.12, 0.28, 0.10),
                     flesh_in=(0.95, 0.94, 0.82), flesh_out=(0.86, 0.90, 0.66)),
    "carrot": dict(L=0.076, r0=0.0145, r1=0.0105, material="carrot", hold=(-0.0135, 0, 0.040, 0.049, 0.040, 0.020),
                   genesis="reports/33_veg_cut_adhesion/adhesion/T4_carrot_strong", skin=(0.85, 0.42, 0.10),
                   flesh_in=(0.98, 0.70, 0.35), flesh_out=(0.95, 0.52, 0.12)),
    # 11-13 추가: 팀원 재료 프리셋에 감자가 없어 우리 configs/materials.yaml potato_lit(문헌값: ρ 1090, E 3 MPa)로 만든다(ν 는 계산을 줄이려 0.35,
    # 파괴 에너지 300 J/m² 는 오이 125·당근 389 사이 가정). 모양은 --shape 로(실제 감자 메쉬).
    "potato": dict(L=0.066, r0=0.024, r1=0.024, material="potato", hold=(-0.0135, 0, 0.040, 0.043, 0.052, 0.020),
                   genesis="reports/33_veg_cut_adhesion/adhesion/T4_potato_strong", skin=(0.66, 0.50, 0.32),
                   flesh_in=(0.96, 0.90, 0.66), flesh_out=(0.94, 0.86, 0.60)),
}
# 11-20 새 식재료(reports/56). 모양은 늘 --shape(scripts/cutting/food/make_food_shapes.py, 겹·껍질 막·색은 그 json). 칼 치수는 오이 Genesis 기록.
# 재료값 근거·가정: docs/food_material_properties_2026-11-19_new_foods.md 6절, reports/56 README 1절.
_NEW = dict(genesis="reports/33_veg_cut_adhesion/adhesion/T4_cucumber_off", skin=(0.9, 0.9, 0.85), flesh_in=(0.95, 0.95, 0.9),
            flesh_out=(0.92, 0.92, 0.86))
for _k, _h in (("garlic", 0.010), ("onion", 0.028), ("green_onion", 0.020), ("luncheon", 0.030), ("sausage", 0.022), ("pork", 0.030)):
    VEG[_k] = dict(_NEW, L=0.05, r0=_h / 2, r1=_h / 2, material=_k, hold=(-0.0135, 0, _h + 0.010, 0.04, 0.04, 0.020))
# 재료(팀원 SoftSolidMaterial 필드). base 의 나머지(판 마찰 0.5·조각끼리 마찰 0.3·시작 균열 2 mm)는 그대로.
NEW_MATS = {
    # 마늘: E = 2.8~15.5 MPa 의 기하 평균(시험법 차이 큼), ρ 실측 1,109~1,146, G_c 가정(감자 300·당근 389 사이), 마찰 = 팀원 당근 대체값
    "garlic": ("carrot", dict(youngs_modulus=6.6e6, poisson_ratio=0.35, density=1120.0, fracture_energy=350.0, knife_friction=0.41,
                              tensile_strength=6.6e5)),
    # 양파: 팀원 프리셋 그대로(1.8 MPa, ρ 1,075, G_c 451, 마찰 0.477)
    "onion": ("onion", dict()),
    # 대파 흰 부분: E 2.44~2.76 MPa(Pan 2023), ρ 조직 계산 1,027, G_c 가정, 마찰 = 오이 값
    "green_onion": ("cucumber", dict(youngs_modulus=2.6e6, poisson_ratio=0.35, density=1027.0, fracture_energy=250.0,
                                     knife_friction=0.36, tensile_strength=2.6e5)),
    # 런천미트: 작은 변형 E 80 kPa(St. Pierre 2024 스팸 칠면조), 인장 파단 16 kPa, ρ 1,055(계산), G_c 20 J/m²(가정 —
    #   솔버 한계 Pi = G_c/(E dx) ≤ 0.157 에 닿음), 이완 = E/2·2 s(가정, tan δ 0.19)
    "luncheon": ("firm_tofu", dict(youngs_modulus=8.0e4, poisson_ratio=0.40, density=1055.0, fracture_energy=20.0, knife_friction=0.30,
                                   relaxed_modulus=4.0e4, relaxation_time=2.0, tensile_strength=1.6e4)),
    # 소시지 속: E 50 kPa(핫도그 44·킬바사 64 사이), 인장 파단 26 kPa, ρ 1,047, G_c 12 J/m²(솔버 한계 13 아래로, 가정), 이완 E/2·2 s.
    #   껍질 막은 --shell_E(입자마다 탄성 배율, 패치 make_warp_mod_scale.py)
    "sausage": ("firm_tofu", dict(youngs_modulus=5.0e4, poisson_ratio=0.40, density=1047.0, fracture_energy=12.0, knife_friction=0.30,
                                  relaxed_modulus=2.5e4, relaxation_time=2.0, tensile_strength=2.6e4)),
    # 생 돼지 등심: ν 0.45, ρ 1,055, 인장 강도 100 kPa(당겨서는 잘 안 찢어짐, 가정), 이완 E/2·1 s(Van Loocke: 점성 몫 약 50 %).
    #   11-21 사용자 결정(reports/58 4절 (가)): E 30 → 75 kPa(압축 문헌 24~75 kPa 의 위 끝). 이 솔버에서 칼 힘은 G_c 가 아니라 E 가 정한다(reports/58 2-5절).
    #   G_c 7.5 → 840 J/m²(문헌값, 11-21 권장대로): Pi = G_c/(E·dx) 6.7 경고는 남지만 끝까지 썰린다.
    #   (11-20 이전 값 E 30 kPa·G_c 7.5. reports/56~58 돼지 판은 E 30 kPa 이고 G_c 는 판마다 --Gc, pork_E75_gc840 만 E 75)
    "pork": ("firm_tofu", dict(youngs_modulus=7.5e4, poisson_ratio=0.45, density=1055.0, fracture_energy=840.0, knife_friction=0.30,
                               relaxed_modulus=3.75e4, relaxation_time=1.0, tensile_strength=1.0e5)),
}

ap = argparse.ArgumentParser()
ap.add_argument("--veg", default="cucumber", choices=sorted(VEG))
ap.add_argument("--genesis_run", default=None, help="칼 기록을 가져올 Genesis 실행(없으면 채소별 기본)")
ap.add_argument("--out", required=True)
ap.add_argument("--E", type=float, default=None)
ap.add_argument("--Gc", type=float, default=None, help="재료 파괴 에너지 G_c [J/m²] 바꿈(11-14 저녁 진단: 팀원 주키니 705 는 문헌·솔버 한계와 안 맞음)")
ap.add_argument("--rho", type=float, default=None, help="재료 밀도 kg/m³ 바꿈(11-19 reports/53: 같은 두께에서 무게만 바꾸는 시험)")
ap.add_argument("--plan2", default=None,
                help="11-19 reports/54 이어 하기: --plan(편썰기)이 끝나면 조각(몸통 뺀 200 입자 이상)을 눕혀 쌓고 이 두께들(mm, 쉼표)로 채썰기. "
                     "옮기기는 물리 계산이 아니라 순간 이동(속도 0·변형 없앰·처음 자리 = 새 자리), 영상에는 옮기는 장면을 끼워 넣음")
ap.add_argument("--stack_x", type=float, default=0.060, help="이어 하기: 쌓은 편의 +x 끝(채썰기 시작 쪽) [m, 우리 좌표]")
ap.add_argument("--stack_from_rest", action="store_true",
                help="11-26 reports/79: 쌓을 편 모양을 지금(변형된) 자리 대신 처음 자리(안 변한 편)를 눕혀서 놓음 — 칼에 붙어 늘어난 편이 "
                     "지금 모양 그대로 옮겨져 쌓은 높이 199 mm 로 퍼진 일(seed 1, 편 4장) 막음. 순간 이동은 어차피 변형을 지움(F = I)")
ap.add_argument("--stack_n", type=int, default=0,
                help="11-26 reports/79: 이어 하기에서 먼저 썬 편 이만큼만 쌓아 채썰기(나머지 편은 그 자리에 둠, 0 = 모두). "
                     "남는 편과 안 겹치게 --stack_x 를 키우고 --side_margin 을 넓힐 것")
ap.add_argument("--stack_gap", type=float, default=1.0e-4, help="이어 하기: 쌓을 때 편 사이 입자 한 간격에 더 띄울 틈 [m]")
ap.add_argument("--settle2_s", type=float, default=0.6, help="이어 하기: 옮긴 뒤 자리 잡는 시간 [s](그동안 칼은 첫 채썰기 자리 위로)")
ap.add_argument("--hold_follow2", type=float, default=0.006, help="이어 하기: 채썰기 때 손 앞 끝 = 칼질 자리 − 이 값 [m]")
ap.add_argument("--lift_clear2", type=float, default=0.0255, help="이어 하기: 채썰기 때 칼 드는 높이 = 쌓은 높이 + 이 값 [m]")
ap.add_argument("--rest2_s", type=float, default=1.0, help="이어 하기: 마지막 채썰기 뒤 멈춤 [s]")
ap.add_argument("--carry_frames", type=int, default=18, help="이어 하기: 영상에만 넣는 옮기는 장면, 조각 하나당 프레임 수(0 이면 안 넣음)")
ap.add_argument("--hold_mode", default="press", choices=["press", "pin", "soft"],
                help="손: press = 잡은 입자 속도만 0(위치는 새어 나감), pin = 잡은 입자 위치를 잡을 때 자리에 고정(고정 경계로 재료가 파고듦), "
                     "soft = 잡을 때 자리로 끄는 용수철 + 감쇠 + 힘 상한(부드러운 손, 11-14 저녁)")
ap.add_argument("--hold_k", type=float, default=1.0e4, help="soft 손 전체 강성 [N/m](가정 10 N/mm)")
ap.add_argument("--hold_fmax", type=float, default=20.0, help="soft 손이 버티는 힘 상한 [N](가정 20 N, 넘으면 미끄러짐)")
ap.add_argument("--hold_tau", type=float, default=0.0,
                help="soft 손: >0 이면 꽉 잡고 버티다 밀린 만큼을 이 시간 상수 [s]로 천천히 되돌림(0 = 임계 감쇠 용수철, 11-14 저녁)")
ap.add_argument("--hold_zclamp", action="store_true", help="soft 손: 위아래는 원래 손처럼 위로 못 뜨게, 용수철은 옆으로만(11-14 저녁 3판)")
ap.add_argument("--dx", type=float, default=2.5e-3)
ap.add_argument("--physics_hz", type=float, default=120.0)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--width", type=int, default=960)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--max_steps", type=int, default=None, help="시험용: 칼 기록의 이 1 ms 단계까지만")
ap.add_argument("--no_fixes", action="store_true")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--face_contact", default="node", choices=["node", "subgrid"], help="팀원 솔버 선택 기능(갈라진 면을 칼날 실제 반두께로)")
ap.add_argument("--fast_box", action="store_true", help="11-23 reports/62: 격자를 재료가 있는 상자만 계산(결과 같음). patches/make_warp_fast_box.py 복사본(<빠른3 솔버 복사본>)이 PYTHONPATH 에 있어야 함")
ap.add_argument("--box_margin", type=int, default=3, help="--fast_box 상자 둘레 여유 [칸](1 ms 동안 입자가 이보다 멀리 가면 틀림 → box_violations)")
ap.add_argument("--fast_diag", action="store_true", help="11-23 reports/62: 진단 _wall_band 를 1 ms 단계 끝에 한 번만(물리 불변, fast-box 복사본)")
ap.add_argument("--fast_launch", action="store_true", help="11-23 reports/62: 서브스텝의 GPU 일 넘기기를 미리 만든 Launch 로 다시 씀(물리 불변, make_warp_fast_launch.py 복사본)")
ap.add_argument("--fast_py", action="store_true", help="11-23 reports/62: 파이썬 쪽 일 줄이기(칼 상태·날 끝 높이 확인·감쇠 조각 고르기, 결과 같음)")
ap.add_argument("--fast", action="store_true", help="11-23 reports/62: 결과 같은 빠르게 하기 전부 = --fast_box --fast_diag --fast_launch --fast_py")
ap.add_argument("--fast_pcbox", action="store_true", help="11-23 reports/63: 격자 지우기를 조각마다 제 상자만(결과 같음). patches/make_warp_fast2.py 복사본(<빠른3 솔버 복사본>) 필요")
ap.add_argument("--fast_fuse", action="store_true", help="11-23 reports/63: P2G 에 칼 옆·틈·도마 닿음 입자 커널을 합침(결과 같음, fast2 복사본)")
ap.add_argument("--fast_sel2", action="store_true", help="11-23 reports/63: 감쇠 조각 고르기 2판(조각 상자를 reduceat 으로, 같은 답)")
ap.add_argument("--fast2", action="store_true", help="11-23 reports/63: --fast + --fast_pcbox --fast_fuse --fast_sel2")
ap.add_argument("--fast_gmask", action="store_true", help="11-23 reports/63: 격자점 계산에서 질량 없는 조각 건너뛰기(결과 같음, <빠른3 솔버 복사본>)")
ap.add_argument("--fast3", action="store_true", help="11-23 reports/63: --fast2 + --fast_gmask. 11-24 사용자 결정으로 기본 켬(--no_fast3 로 끔)")
ap.add_argument("--no_fast3", action="store_true",
                help="11-24: 기본 빠른3 을 끄고 원래 코드로(속도 비교·확인용). --fast·--fast2 를 직접 주면 그 단계만 켜짐(기본 빠른3 은 안 켬)")
ap.add_argument("--fast_poison", action="store_true", help="11-23 reports/63 시험용: 상자 밖 독 넣기·상자 밖 0 검사(patches/make_warp_poison.py 복사본 <빠른3 솔버 복사본>)")
ap.add_argument("--pos_jitter", type=float, default=0.0, help="11-23 reports/63 시험용: 시작 전 입자 자리에 표준편차 이만큼[m] 무작위 흔들기(0 = 그대로). 갈림길이 반올림 수준 차이로 뒤집히는지 보려고(예 1e-7)")
ap.add_argument("--jitter_seed", type=int, default=0, help="--pos_jitter 의 난수 씨앗")
ap.add_argument("--jitter_step", type=int, default=0, help="--pos_jitter 를 넣는 1 ms 단계(0 = 시작 전). 갈림 바로 앞에 넣어 자리 잡기 감쇠에 묻히지 않게")
ap.add_argument("--wp_det", type=int, default=0, choices=[0, 1, 2], help="11-23 reports/63 시험용: Warp 결정적 모드(wp.config.deterministic). 1 = RUN_TO_RUN(같은 GPU 에서 원자 덧셈을 (대상, 스레드) 순서로 모아 더해 비트까지 같은 결과), 2 = GPU_TO_GPU. 0 = 끔(원래). 느려짐")
ap.add_argument("--wp_nofma", type=int, default=0, choices=[0, 1], help="11-23 reports/63 시험용: 1 = 솔버·보정 커널 모듈의 곱셈·덧셈 묶기(FMA, Warp fuse_fp)를 끔. 0 = 원래(묶음)")
ap.add_argument("--cfl", type=float, default=None, help="11-23 reports/62: 시간 간격 = cfl × dx / 파동 속도(솔버 기본 0.35)")
ap.add_argument("--knife_floor", type=float, default=None,
                help="칼 날 끝 높이 하한 [m, 도마 면 기준]. 팀원 솔버는 안 갈라진 곳에서 칼날을 dilation(0.75칸=1.9 mm) 두껍게 쳐서, 날 끝이 "
                     "그보다 도마에 가까우면 둥근 날 끝이 도마 속으로 파고들어 마지막 층을 짓누르다 잘리는 순간 조각을 차낸다. "
                     "dilation 안이면 남은 층은 그냥 갈라지므로(_advance_crack 의 pinch) 이 높이에서 멈춰도 끝까지 잘린다(11-13)")
ap.add_argument("--adh_release", default="top", choices=["top", "none"],
                help="top = 칼이 맨 위에서 멈추는 단계(top·move·rest)에 접착을 꺼 붙은 조각을 떨어뜨림(규칙). none = 끄지 않음 → 조각이 칼에 실려 "
                     "다음 칼질 자리로 가 칼등 위로 넘어가고 칼날이 그 조각을 뚫는다(11-13 tune/cucumber_film_split2)")
ap.add_argument("--bounce_damp", type=float, default=0.0,
                help="[1/s] 칼에 안 붙은 조각이 도마 면 근처에서 위로 튀는 속도 줄이기(규칙). 팀원 오이는 거의 순수 탄성이라 칼에서 떨어진 조각이 "
                     "고무공처럼 튐(11-13, 54 mm 에서 떨어져 58 mm 까지). 0 = 끔")
ap.add_argument("--side_margin", type=float, default=0.06, help="계산 상자 옆 여백 [m](넘어진 조각이 벽 띠에 안 닿게)")
ap.add_argument("--top_margin", type=float, default=None,
                help="계산 상자 천장 = 재료 윗면 + 이 값 [m]. 기본 0.035, 접착을 켜면 0.09(칼에 붙은 조각이 칼을 따라 55 mm 까지 올라가는데 "
                     "0.035 면 천장 벽 띠(4.5칸)에 닿아 붙잡힌다, 11-13)")
ap.add_argument("--blade_split", action="store_true",
                help="칼날 너머 조각끼리 맞닿지 않게(복사본 솔버 patches/make_warp_blade_split.py). 끄면 몸통이 칼 너머 조각을 "
                     "격자 2칸(5 mm) 밖에 붙잡아 둬서 칼 면에 3~4 mm 빈틈이 생긴다(11-13)")
ap.add_argument("--gap_contact", type=float, default=0.0,
                help="조각끼리 실제로 닿은 곳만 맞닿음(복사본 patches/make_warp_gap_contact.py): 격자점에서 두 조각 질량 중심 차가 이 값(칸) "
                     "이하일 때만. 0 = 끔(원래: 3칸 = 5 mm 떨어져도 맞닿음). 11-14 권장 1.0")
ap.add_argument("--face_tan", action="store_true",
                help="칼 면 가까이 조각끼리는 칼 면을 따라서만 민다(복사본 patches/make_warp_face_tangent.py, --gap_contact 필요). "
                     "끄면 칼에 붙은 조각이 다음 조각에 밀려 칼 바깥으로 기울어 떨어진다(11-14 오후)")
ap.add_argument("--knife_bevel", type=float, default=None,
                help="칼 날 쐐기 높이 [m](날 끝 두께 → 칼등 두께로 두꺼워지는 높이). 기본은 Genesis T4 의 6 mm(1.5 mm 판에 6 mm 쐐기). "
                     "0.045 = 등에서 날 끝까지 고르게 얇아지는 식칼(11-14 오후: 좁은 채가 쐐기에 밀려 굴러 떨어지던 것)")
ap.add_argument("--release_outer", action="store_true",
                help="칼을 드는 동안(withdraw) 칼 바깥쪽(잘려 나간 쪽) 조각은 칼 옆 감쇠(sep_damp)에서 뺀다. "
                     "끄면 칼에 붙은 작은 조각이 처음 0.3 초 동안 감쇠에 붙잡혀 못 따라온다(11-14 오후)")
ap.add_argument("--end_damp", type=float, default=0.0,
                help="칼질 끝 감쇠 [1/s]: 날 끝이 도마 위 --end_h 아래로 온 뒤 dwell 끝까지 칼 옆 감쇠(기본 300/s)를 이 값으로(warp_fixes). "
                     "0 이면 끔. 끄면 단단한 채소(당근 6 MPa)는 마지막 남은 층이 갈릴 때 막 썬 조각이 칼 면에서 튕겨 나가 안 붙는다(11-14 밤, reports/45)")
ap.add_argument("--end_h", type=float, default=0.008, help="칼질 끝 감쇠를 켜는 날 끝 높이 [m, 도마 면 기준]")
ap.add_argument("--dense_below", type=float, default=0.0,
                help="진단(11-14 밤): 날 끝이 도마 위 이 높이 [m] 아래인 cut·dwell·withdraw 동안 --dense_every 단계(1 ms)마다 입자 위치·속도·라벨·균열 앞을 "
                     "dense.npz 로 저장. 0 이면 끔")
ap.add_argument("--dense_every", type=int, default=2)
ap.add_argument("--pinch_extra", type=float, default=0.0,
                help="바닥 가르기(pinch) 문턱을 dilation(1.25 mm) + 이 값 [m] 으로(복사본 patches/make_warp_pinch.py). 0 = 원래. "
                     "날 끝이 도마 두 칸 위 격자점 줄을 누르기 전에 남은 층을 갈라, 눌려 있던 재료가 막 떨어진 조각을 튕겨 내지 않게 한다(11-14 밤, reports/45)")
ap.add_argument("--dilation", type=float, default=None,
                help="팀원 솔버 SimConfig.dilation [칸]: 안 갈라진 곳에서 칼날을 두껍게 치는 양(기본 0.75칸 = 1.25 mm). 줄이면 막 썬 조각을 덜 밀어 벌림(11-14 밤 시험)")
ap.add_argument("--spine_topple", type=float, default=0.0,
                help="칼등 위 넘어뜨리기 [rad/s](11-19 저녁, reports/55, warp_fixes): 조각 무게중심이 칼등보다 높고 칼 평면 가까이(--spine_near)면 "
                     "가장 낮은 점을 축으로 칼에서 먼 쪽으로 넘어지게 이 각속도를 한 번 더함(규칙). 0 = 끔")
ap.add_argument("--spine_near", type=float, default=0.006, help="칼등 위 넘어뜨리기: 무게중심이 칼 평면에서 이 거리 [m] 안일 때만")
ap.add_argument("--spine_below", type=float, default=0.05,
                help="칼등 위 넘어뜨리기: 칼등보다 낮은 입자가 이 비율 이하일 때만(거의 다 칼등 위 — 아랫부분이 칼 면에 붙은 편은 안 건드림)")
ap.add_argument("--spine_cool", type=float, default=0.1, help="칼등 위 넘어뜨리기: 같은 조각에 다시 걸 때까지 기다리는 시간 [s]")
ap.add_argument("--spine_max", type=int, default=3, help="칼등 위 넘어뜨리기: 조각마다 최대 횟수")
ap.add_argument("--end_scope", default="pair", choices=["pair", "all"],
                help="칼질 끝 감쇠 대상: pair = 지금 칼질이 가르는 짝(몸통·막 썬 조각)만(기본), all = 칼 옆 감쇠 대상 전부(첫 시험 E1)")
ap.add_argument("--reside", action="store_true",
                help="균열이 지나가기 전까지는 칼 평면의 실제 쪽으로 조각을 다시 나눔(복사본 patches/make_warp_reside.py). "
                     "끄면 칼질 시작 뒤 재료가 움직여 칼이 채를 가로지르고 채가 칼에 꿰인다(11-14 밤)")
ap.add_argument("--knife_side", action="store_true",
                help="칼 접촉의 쪽을 격자점마다 입자로 정함(복사본 patches/make_warp_knife_side.py). 끄면 칼에 붙어 칼등을 넘은 조각을 칼날이 뚫는다(11-13)")
ap.add_argument("--knife_roll_deg", type=float, default=0.0,
                help="칼 기울기(칼날 면을 날 끝 축으로 돌림) [°]. 음수 = 칼등이 +x(잘려 나가는 조각) 쪽으로 — 조각이 쐐기꼴로 잘리고 위쪽이 "
                     "바깥으로 밀린다(사람 칼질처럼 수직이 아님, 11-13)")
ap.add_argument("--knife_height", type=float, default=None, help="칼날 높이 [m] 덮어쓰기(Genesis 칼 35 mm, 보통 식칼 40~50 mm)")
ap.add_argument("--plan", default=None,
                help="칼질 계획: 끝에서부터 조각 두께 mm 를 쉼표로(예 '3,3,6'). 주면 Genesis 칼 기록 대신 같은 빠르기(썰기 40 mm/s, "
                     "들기 50 mm/s, 맨 위 --top_s 초)로 칼 경로를 만든다")
ap.add_argument("--top_s", type=float, default=0.4, help="--plan: 칼을 맨 위에서 멈추는 시간 [s]")
ap.add_argument("--rest_s", type=float, default=0.3, help="--plan: 마지막에 기다리는 시간 [s]")
ap.add_argument("--lift", type=float, default=0.0555, help="--plan: 칼을 드는 높이(날 끝, 도마 면 기준) [m]")
ap.add_argument("--lift_clear", type=float, default=0.0255,
                help="--plan: 칼을 적어도 재료 윗면 + 이만큼 듦 [m](드는 높이 = max(--lift, 윗면 + 이 값)). 11-14: 영상을 줄이려 0.015")
ap.add_argument("--length", type=float, default=None, help="채소 길이 [m] 덮어쓰기(짧게 하면 빠름)")
ap.add_argument("--hold_gap", type=float, default=0.009, help="--plan: 손 상자 앞 끝 = 마지막 칼질 − 이 값 [m]")
ap.add_argument("--hold_follow", type=float, default=None,
                help="--plan: 손이 칼질마다 따라 물러남 — 칼질 k 의 다가가기 시작에 손 상자 앞 끝 = 칼질 k − 이 값 [m]으로 다시 잡음"
                     "(채썰기 시연: 세운 편이 손 밖에서 부채처럼 펼쳐지며 쓰러진 것 막기, 11-13)")
ap.add_argument("--wet", type=float, default=None, help="기본 1(소시지 0.3, FOOD_DEFAULTS). 접착(물막) 세기 배율: 흡인 p_a·떼기 점성 c_n·미끄럼 점성 c_t·미끄럼 한계 tau·sigma 에 모두 곱함(수분 많을수록 큼). 11-19 reports/52: 에피소드마다 로그정규(가운데 0.837, σ 0.228)로 뽑기 추천")
ap.add_argument("--max_pieces", type=int, default=None, help="솔버가 다룰 조각 수 상한(격자 메모리 ∝ 이 값)")
ap.add_argument("--knife_friction", type=float, default=None,
                help="칼–재료 쿨롱 마찰 덮어쓰기(재료 프리셋 knife_friction, 오이 0.36). 썰기·붙음 모두에 걸림(11-17 Tier 0 재현)")
ap.add_argument("--knife_friction_post", type=float, default=None,
                help="칼질(cut) 단계 밖(dwell·withdraw·top·move·approach·rest)에서 쓸 칼 마찰 — 썰기는 원래 값 그대로. "
                     "11-17 Tier 0 재현(reports/46)에서 붙음의 마찰 몫을 가르려고 넣음, reports/47 에서 다음 칼질 전에 원래 값으로 되돌리게 고침"
                     "(전에는 한 번 바꾸면 끝까지 그 값). 0 이면 칼에 붙은 조각을 칼 격자 마찰이 받치지 않음(흡인 × 마찰 이중 계산·격자 의존 빼기)")
ap.add_argument("--adh_anchor", action="store_true",
                help="11-17(reports/47): 접착 구역을 칼 면이 아니라 조각 자신의 칼 쪽 면(입자 거리 10 백분위)에 맞춤 — 칼–격자 어긋남이 만든 "
                     "가짜 틈(−0.2~+0.4 mm)이 τ 를 흔들지 않게(warp_fixes._adh_gap_*)")
ap.add_argument("--adh_anchor_hi", type=float, default=0.6e-3, help="--adh_anchor 로 구역을 옮기는 최대 거리 [m](더 멀면 떨어지는 중으로 봄)")
ap.add_argument("--contact_friction", type=float, default=None,
                help="조각끼리 마찰 덮어쓰기(재료 프리셋 contact_friction, 오이 0.3) — 11-17 저녁 reports/47 진단")
ap.add_argument("--kerf_slip", action="store_true",
                help="날 끝 바로 아래 열린 틈(날 끝 아래 3칸·칼 평면 ±(날 끝 반두께 + 2칸))의 조각끼리 맞닿음은 미끄럼 마찰 0"
                     "(복사본 patches/make_warp_kerf_slip.py, 11-17 저녁 reports/47)")
ap.add_argument("--kerf_depth", type=float, default=3.0,
                help="--kerf_slip 구역 깊이(날 끝 아래, 칸). 3 칸이면 칼을 들 때 처진 무거운 조각(4 mm) 밑동이 구역을 벗어나 다시 붙잡혔다 → 30 이면 도마까지")
ap.add_argument("--pair_adh", action="store_true",
                help="11-18(reports/48): 맞닿은 조각끼리 물막 접착(떨어짐 = 흡인 p_a + 떼기 점성, 미끄럼 = τ + 미끄럼 점성, 칼 접착과 같은 값 × "
                     "--pair_adh_scale). 몸통(가장 큰 조각)은 빼고(--pair_adh_body 면 넣음). 복사본 patches/make_warp_pair_adhesion.py + --gap_contact 필요")
ap.add_argument("--pair_adh_scale", type=float, default=1.0, help="--pair_adh 세기 배율(칼 접착 값에 곱함)")
ap.add_argument("--pair_adh_body", action="store_true", help="--pair_adh 에 몸통(가장 큰 조각)도 넣음")
ap.add_argument("--pair_adh_face_cos", type=float, default=0.0,
                help="> 0 이면 맞닿음 방향이 칼 평면 법선과 이 cos 이상인 곳(젖은 단면끼리)만 붙임 — 껍질 테두리끼리는 뺌. 0 = 모든 맞닿음")
ap.add_argument("--pair_adh_gap", type=float, default=None, help="--pair_adh 닿음 판정 거리 [칸] (기본 = --gap_contact 값)")
ap.add_argument("--shape", default=None,
                help="실제 모양: scripts/cutting/food/veg_shape_prep.py 결과 앞부분 경로(예 data/veg/shapes/cucumber_k40). 주면 원기둥 대신 그 메쉬 안을 입자로 채움")
ap.add_argument("--planks", action="store_true", help="--shape 의 편 경계(plank_edges)로 처음부터 조각을 나눔(채썰기 시연: 편을 모아 세운 것)")
ap.add_argument("--no_layers", action="store_true", help="11-20: --shape json 의 겹(layers)을 무시하고 한 덩어리로(양파·대파 비교용)")
ap.add_argument("--shell_E", type=float, default=None,
                help="11-20: --shape json 의 껍질 막(shell) 입자 영률 [Pa]. 입자마다 탄성 배율 패치(patches/make_warp_mod_scale.py)가 든 "
                     "복사본 필요, 시간 간격을 1/√배율로 줄임. 소시지 기본 9e6(11-21 사용자 결정, FOOD_DEFAULTS), 0 = 껍질 막 끔")
ap.add_argument("--shell_gc_mul", type=float, default=None,
                help="기본 1(소시지 90, FOOD_DEFAULTS). 11-20: 껍질 막 두께 띠(칼질 시작 때 윗면에서 아래로)를 지나는 균열의 G_c 배율(껍질 질김). "
                     "patches/make_warp_shell_toughness.py 가 든 복사본 필요")
ap.add_argument("--shell_gc_mode", default=None, choices=["band", "particle"],
                help="기본 band(소시지 particle, FOOD_DEFAULTS). 11-20 저녁(reports/58): band = 윗면에서 shell 두께 띠만(1판), particle = 칼날 칸마다 껍질 입자가 차지한 높이 구간"
                     "(옆면·아래 껍질 포함, patches/make_warp_shell_toughness2.py → <빠른3 솔버 복사본>)")
ap.add_argument("--region_E_mul", type=float, default=None,
                help="11-20 저녁(reports/58): 처음 자리 x > --region_xmin 인 입자만 영률 배율(대파 초록 잎을 무르게). 입자마다 탄성 배율 패치 필요")
ap.add_argument("--region_xmin", type=float, default=0.0, help="--region_E_mul 을 걸 처음 자리 x 하한 [m, 우리 좌표]")
ap.add_argument("--tensile", type=float, default=None,
                help="11-20 저녁(reports/58): 재료 인장 강도 tensile_strength [Pa] 바꿈. 솔버의 찢김(tearing) 문턱 — G_c 를 올려도 여기서 먼저 찢기면 질김이 안 늘어남")
ap.add_argument("--relaxed_E", type=float, default=None,
                help="11-21 새벽(reports/58): 재료 이완 영률 relaxed_modulus [Pa] 바꿈. --E 를 바꿀 때 이완 비율(돼지 E/2)을 지키려고")
ap.add_argument("--saw_sever", default=None,
                help="11-21 reports/59: 'H_mm,L_mm' — 도마 위 H mm 질긴 아래층은 날이 그 안에서 날 방향으로 L mm 미끄러져야 끊김(톱질 규칙). "
                     "솔버 복사본 <빠른3 솔버 복사본>(patches/make_warp_saw_sever.py) 필요. 'off' = 끔. "
                     "생 돼지는 11-21 사용자 결정으로 기본 켬(FOOD_DEFAULTS)")
ap.add_argument("--cut_speed", type=float, default=0.04,
                help="11-25 reports/76: 칼질(cut) 단계 내려가는 빠르기 [m/s](기본 0.04 = Genesis T4 와 같음). 밀며 썰기 ξ 는 이 빠르기 기준이라 "
                     "같은 ξ 로 견주려면 --saw_hz 도 같은 비율로")
ap.add_argument("--bottom_saw_s", type=float, default=None,
                help="11-21 reports/59: 칼질마다 날 끝이 바닥(도마 위 0.5 mm)에 닿은 뒤 이 시간 [s] 머물며 --saw_amp·--saw_hz 로 앞뒤로 밂(bsaw 단계). "
                     "기본 0(생 돼지는 1)")
ap.add_argument("--pull_check", default=None,
                help="11-21 reports/60 확인 동작: 'D_mm,V_mm_s' — 칼질이 다 끝난 뒤 손(보이지 않음)이 잡은 덩이를 -x(편 반대쪽)로 D mm 를 V mm/s 로 끌고 "
                     "0.8초 둔다(pull·pullrest 단계). 편이 덩이와 아직 이어져 있으면 따라오고, 다 잘렸으면 제자리에 남는다. --plan·--hold_mode press 에서만")
ap.add_argument("--saw_region", default=None, choices=["all", "pork_tissue"],
                help="11-21 reports/60: 질긴 아래층을 둘 자리. all = 칼날 폭 전체(reports/59 첫 판), "
                     "pork_tissue = 생 돼지의 지방 띠·막 + 살 속 근막 줄만(scripts/cutting/food/food_regions.py). 생 돼지 기본 pork_tissue")
ap.add_argument("--no_tearing", action="store_true",
                help="11-20 저녁(reports/58): 덜 잘린 곳이 인장 강도에서 저절로 찢기는 규칙(SimConfig.tearing)을 끔 — 균열은 칼의 일(G_c 셈)으로만 열림")
ap.add_argument("--saw_amp", type=float, default=None,
                help="11-20: 칼질(cut) 동안 칼을 날 방향(y)으로 밀고 당기는 진폭 [m](0 이면 수직 누르기만). 기본 0(생 돼지는 0.010)")
ap.add_argument("--saw_hz", type=float, default=None, help="11-20: 밀고 당기기 빈도 [Hz]. 기본 1.5(생 돼지는 2.0)")
ap.add_argument("--force_cap", default="none",
                help="11-24 reports/65: 칼 힘 한계. 숫자 [N] 또는 elrobot_console(12.1)·elrobot_motor(18.6)·none(기본, 지금과 똑같이 돎). "
                     "칼질(cut) 중 위쪽 힘이 한계에 다가가면 그 1 ms 는 덜 내려가고 넘으면 멈춤(scripts/cutting/food/force_cap.py). "
                     "밀며 썰기(--saw_amp)는 시간에 따라 계속. --plan 경로에서만")
ap.add_argument("--force_mode", default="admit", choices=["admit", "hold"],
                help="힘 한계 방식: admit = 내려가는 몫 (한계−힘)/(band×한계)(기본, 천천히 눌러 밀며 썰기 이득이 실제처럼), hold = 넘으면 0·아니면 1")
ap.add_argument("--force_band", type=float, default=0.2, help="admit: 힘이 한계의 (1 − 이 값)부터 줄이기 시작")
ap.add_argument("--force_filter_ms", type=int, default=5, help="힘 한계: 최근 몇 ms 평균 힘으로 볼지(솔버 힘의 톱니 거르기)")
ap.add_argument("--stall_s", type=float, default=2.0, help="힘 한계: 이 시간 동안 --stall_dz_mm 만큼도 안 내려가면 그 칼질은 '멈춤'")
ap.add_argument("--stall_dz_mm", type=float, default=0.2, help="힘 한계: 멈춤 판정 깊이 [mm]")
ap.add_argument("--stall_max_s", type=float, default=10.0, help="힘 한계: 칼질 하나가 계획보다 이만큼 늦어지면 '멈춤' [s]")
ap.add_argument("--no_og", action="store_true", help="빠른 맞춤용: OmniGibson 없이 솔버만(그림 없음, pip Warp)")
ap.add_argument("--adhesion", default="none", choices=["none", "film", "film2", "film3"],
                help="칼 접착: film = Genesis ④ 물막 판(warp_fixes.ADHESION_FILM), film2 = 11-13 맞춘 판(ADHESION_FILM2, 보통 수분 = --wet 1)")
ap.add_argument("--adh_layers", type=float, default=None,
                help="접착 구역 두께 = 칼 면 접촉 거리 + 이 값 × 입자 간격(주면 r0·r1 대신). 물막은 맞닿은 면에만 작용 → 붙는 힘이 면적에 비례(11-13). "
                     "r1 5 mm 면 얇은 조각은 통째로 구역에 들어 붙는 힘이 부피(무게)에 비례해 두께 차이가 안 난다")
for _k in ("sigma", "tau", "c_n", "c_t", "p_a", "r0", "r1", "min_kw"):
    ap.add_argument(f"--adh_{_k}", type=float, default=None, help=f"접착 {_k} 덮어쓰기(Pa, Pa·s/m, m)")
args = ap.parse_args()
CUT_SPEED = float(args.cut_speed)  # 11-25: make_plan 의 칼질 빠르기


def _fast3_overlay():
    """11-24 사용자 결정("네, 기본으로"): 빠른3(결과 같은 속도 고침, reports/62·63)을 기본으로.

    솔버는 PYTHONPATH 의 복사본에서 불러온다. 원본 복사본(<빠른3 솔버 복사본>·overlay7·overlay8 …)이 있으면 같은 짝의
    <원본>_fast3(patches/build_fast3_overlay.sh 가 만듦)으로 sys.path 를 바꾼다. 단, 빠른 복사본의 도장(FAST3_BASE.json)이
    지금 원본 .py 파일들과 다르면(원본이 그 뒤 고쳐짐) 바꾸지 않고 원본 그대로 돈다(결과는 같고 느릴 뿐) → 경고.
    돌려주는 값: (기본으로 켰나, 쓴 빠른 복사본 경로 또는 None, 알림 글)."""
    import hashlib
    import re
    if args.no_fast3 or args.fast or args.fast2:
        return False, None, None
    if not args.no_og:  # 11-24 reports/64: 같은 프로세스 OmniGibson 은 Isaac Sim 의 Warp 1.8.2 → 빠른 복사본(Warp 1.19 용, warp._src) 못 불러옴
        return False, None, "OmniGibson 같은 프로세스 판(--no_og 없음)은 Isaac Sim 의 Warp 1.8 이라 빠른3 을 안 씀(원래 코드). 데이터는 --no_og 물리 + og_mesh_render 로"
    args.fast3 = True
    note = None
    for i, p in enumerate(list(sys.path)):
        if not p:
            continue
        base = Path(p)
        if not re.fullmatch(r"simfoundry_overlay\d*", base.name):
            continue
        cand = base.with_name(base.name + "_fast3")
        stamp = cand / "FAST3_BASE.json"
        if not stamp.exists():
            note = f"{cand} 없음 → 원본 {base} 그대로(만들기: bash patches/build_fast3_overlay.sh {base})"
            break
        st = json.loads(stamp.read_text())
        now = {str(f.relative_to(base)): hashlib.sha256(f.read_bytes()).hexdigest()
               for f in sorted(base.rglob("*.py")) if "__pycache__" not in f.parts}
        if Path(st.get("base", "")).resolve() != base.resolve() or st.get("base_sha256") != now:
            note = (f"원본 {base} 가 빠른 복사본을 만든 뒤 바뀜 → 원본 그대로(다시 만들기: bash patches/build_fast3_overlay.sh {base})")
            break
        sys.path[i] = str(cand)
        return True, str(cand), None
    return True, None, note


import sys  # noqa: E402

FAST3_AUTO, FAST3_PATH, FAST3_NOTE = _fast3_overlay()
if FAST3_PATH:
    print(f"  빠른3 기본(11-24): 솔버 {FAST3_PATH}", flush=True)
elif FAST3_NOTE:
    print(f"  [{'경고' if FAST3_AUTO else '알림'}] 빠른3 기본: {FAST3_NOTE}", flush=True)
V = VEG[args.veg]
# 11-21 사용자 결정(reports/59 4절 → reports/60): 생 돼지는 '도마 위 질긴 아래층은 앞뒤로 밀어야 끊김'을 기본으로 켬 — 층 1.7 mm·미끄러짐 40 mm,
#   일부 부위(지방 띠·막 + 살 속 근막 줄)에만. 바닥 톱질 1초와 칼질 중 밀고 당기기(±10 mm, 2 Hz)도 함께(이게 없으면 그 자리가 안 끊김).
# 11-21 저녁 사용자 결정("소시지 9mpa로 해줘", reports/59 2-3절): 껍질 소시지는 껍질 영률 9 MPa·껍질 질김 ×90(칸마다 껍질 입자 구간)·접착 0.3 이 기본.
#   솔버는 overlay7 이상(overlay8 도 됨). reports/58 sausage_p90 은 껍질 3 MPa, reports/59 sausage_p90_sE9 가 이 기본과 같은 판.
FOOD_DEFAULTS = {"pork": dict(saw_sever="1.7,40", saw_region="pork_tissue", bottom_saw_s=1.0, saw_amp=0.010, saw_hz=2.0),
                 "sausage": dict(shell_E=9.0e6, shell_gc_mul=90.0, shell_gc_mode="particle", wet=0.3)}
for _k, _v in FOOD_DEFAULTS.get(V.get("material"), {}).items():
    if getattr(args, _k) is None:
        setattr(args, _k, _v)
        print(f"  {V['material']} 기본값: --{_k} {_v}", flush=True)
if str(args.saw_sever).lower() in ("off", "0", "none", ""):
    args.saw_sever = None
args.bottom_saw_s = args.bottom_saw_s or 0.0
args.saw_amp = args.saw_amp or 0.0
args.saw_hz = args.saw_hz or 1.5
args.saw_region = args.saw_region or "all"
args.wet = 1.0 if args.wet is None else args.wet
args.shell_gc_mul = 1.0 if args.shell_gc_mul is None else args.shell_gc_mul
args.shell_gc_mode = args.shell_gc_mode or "band"
if args.saw_sever and args.bottom_saw_s > 0 and args.saw_amp <= 0:
    raise SystemExit("--bottom_saw_s 는 --saw_amp(앞뒤로 밀기 진폭)가 있어야 날이 미끄러짐")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 11-24: force_cap(아래에서 warp_fixes 도 같은 폴더)
from force_cap import ForceGate, parse_cap  # noqa: E402
FORCE_CAP_N = parse_cap(args.force_cap)  # None = 한계 없음(지금과 똑같은 코드 길)
if FORCE_CAP_N is not None and not args.plan:
    raise SystemExit("--force_cap 은 --plan(우리 칼 경로)에서만 — Genesis 칼 기록 재생에는 칼질 단계 표가 달라 안 씀")
SAW_TAB = {}  # 칼질 번호 → 밀며 썰기 줄·사인·더하기 전 값(_add_saw, 힘 한계 판만 씀)
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
t_wall = time.time()

gr = Path(args.genesis_run or V["genesis"])
if (gr / "log.npz").exists():  # Genesis 칼 기록 재생(작성자 저장소의 기록)
    glog = np.load(gr / "log.npz", allow_pickle=True)
    gmeta = json.loads(str(np.load(gr / "frames.npz")["meta"]))
    Qg, PH = glog["q"].astype(float), glog["phase"]
elif args.plan:  # 우리 칼 경로(--plan): 칼 치수·카메라·색 같은 메타만 있으면 됨 → 이 폴더의 genesis_meta/<기록 이름>.json
    gmeta = json.loads((Path(__file__).resolve().parent / "genesis_meta" / f"{gr.name}.json").read_text())
    gmeta["marks"] = []
    Qg, PH = np.zeros((1, 4)), np.array(["start"])  # 아래 --plan 이 경로를 새로 만듦
else:
    raise SystemExit(f"칼 경로가 없음: --plan 을 주거나 Genesis 칼 기록 폴더({gr})를 --genesis_run 으로")
if args.length is not None:
    V = dict(V, L=float(args.length))
SH = json.loads(Path(args.shape + ".json").read_text()) if args.shape else None
if SH:  # 실제 모양: 경계 상자(우리 좌표 = 모양 좌표)
    (X0, Y0, _), (X1, Y1, ZT) = SH["bounds"]
    V = dict(V, L=X1 - X0)
else:  # 원기둥/원뿔대: 가운데 0, 바닥 z 0
    X0, X1, Y0, Y1, ZT = -V["L"] / 2, V["L"] / 2, -V["r0"], V["r0"], 2 * V["r0"]
if args.knife_height is not None:
    gmeta["knife"] = dict(gmeta["knife"], height=float(args.knife_height))


# 칼질(cut) 단계 내려가는 빠르기 CUT_SPEED [m/s] 는 args 를 읽은 바로 뒤에 --cut_speed 로 정함(기본 0.04, 11-25 reports/76)
def make_plan(th_mm, x_end, z_top, lift, hover=0.0385, top_s=0.4, rest_s=0.3, settle_ms=150, bottom_saw_ms=0.0):
    """Genesis T4 와 같은 빠르기의 칼 경로(1 ms). 칼질 k: approach(50 mm/s) → cut(40 mm/s, 재료 위 3.4 mm 에서 도마 위 0.5 mm 까지)
    → dwell 50 ms → withdraw(50 mm/s, lift 까지) → top(top_s) → 다음 자리로 move(80 mm/s)."""
    q, ph = [], []

    def seg(name, a, b, ms):
        ms = max(1, int(round(ms)))
        for j in range(ms):
            q.append(a + (b - a) * (j + 1) / ms)
            ph.append(name)

    cuts, acc = [], 0.0
    for t in th_mm:
        acc += t / 1000.0
        cuts.append(x_end - acc)
    cur = np.array([cuts[0], 0.0, max(hover, z_top + 0.0085), 0.0])
    q.append(cur.copy()); ph.append("start")
    seg("settle", cur, cur, settle_ms)
    for kk, cx in enumerate(cuts):
        if kk > 0:
            b = np.array([cx, 0.0, lift, 0.0])
            seg(f"move{kk}", cur, b, abs(b[0] - cur[0]) / 0.08 * 1000)
            cur = b
        b = np.array([cx, 0.0, z_top + 0.0035, 0.0])
        seg(f"approach{kk}", cur, b, abs(cur[2] - b[2]) / 0.05 * 1000)
        a, b = b, np.array([cx, 0.0, 0.0005, 0.0])
        seg(f"cut{kk}", a, b, (a[2] - b[2]) / CUT_SPEED * 1000)  # 11-25: 기본 0.04 m/s(--cut_speed)
        if bottom_saw_ms > 0:  # 11-21 reports/59: 바닥에 머물며 앞뒤로 밀기(y 흔들기는 아래 saw 에서)
            seg(f"bsaw{kk}", b, b, bottom_saw_ms)
        seg(f"dwell{kk}", b, b, 50)
        a, b = b, np.array([cx, 0.0, lift, 0.0])
        seg(f"withdraw{kk}", a, b, (b[2] - a[2]) / 0.05 * 1000)
        seg(f"top{kk}", b, b, top_s * 1000)
        cur = b
    seg("rest", cur, cur, rest_s * 1000)
    return np.array(q), np.array(ph), cuts


if args.plan:
    th = [float(v) for v in args.plan.split(",")]
    Qg, PH, plan_cuts = make_plan(th, X1, ZT, max(args.lift, ZT + args.lift_clear), top_s=args.top_s, rest_s=args.rest_s,
                                  bottom_saw_ms=args.bottom_saw_s * 1000)
    def _add_saw(ks):  # 11-20: 칼질 동안 날 방향(y)으로 밀고 당기기, 칼질이 끝나면 머무는(dwell) 동안 0 으로 되돌림
        for _kk in ks:
            _ic = np.flatnonzero((PH == f"cut{_kk}") | (PH == f"bsaw{_kk}"))  # 11-21: 바닥 톱질(bsaw)도 같은 사인으로 이어서
            _yc = args.saw_amp * np.sin(2 * math.pi * args.saw_hz * (np.arange(len(_ic)) + 1) * 1e-3)
            _iw = np.flatnonzero(PH == f"dwell{_kk}")
            # 11-24 reports/65: 힘 한계(--force_cap)가 칼을 늦추면 밀며 썰기를 시간에 따라 이어 가려고 더하기 전 값과 사인을 남김
            SAW_TAB[_kk] = dict(ic=_ic, yc=_yc, iw=_iw, base_ic=Qg[_ic, 1].copy(), base_iw=Qg[_iw, 1].copy())
            Qg[_ic, 1] += _yc
            Qg[_iw, 1] += _yc[-1] * (1 - (np.arange(len(_iw)) + 1) / len(_iw))

    if args.saw_amp > 0:
        _add_saw(range(len(plan_cuts)))
    if args.pull_check:  # 11-21 reports/60 확인 동작: 칼은 마지막 자리 위에 든 채로, 손이 덩이를 -x 로 끔
        if args.hold_mode != "press":
            raise SystemExit("--pull_check 는 --hold_mode press 에서만")
        PULL_D, PULL_V = (float(v) / 1000.0 for v in args.pull_check.split(","))
        _n1, _n2 = int(round(PULL_D / PULL_V * 1000)), 800
        Qg = np.concatenate([Qg, np.repeat(Qg[-1:], _n1 + _n2, axis=0)])
        PH = np.concatenate([PH, np.array(["pull"] * _n1 + ["pullrest"] * _n2)])
        print(f"  확인 동작: 칼질 뒤 손이 덩이를 -x 로 {PULL_D * 1e3:.0f} mm({PULL_V * 1e3:.0f} mm/s, {_n1} ms) 끌고 0.8초 둠", flush=True)
    x_max = plan_cuts[-1] - args.hold_gap  # 손 상자: 채소 왼쪽 끝 ~ 마지막 칼질 − hold_gap
    V = dict(V, hold=((x_max + X0) / 2, (Y0 + Y1) / 2, V["hold"][2], x_max - X0, (Y1 - Y0) + 0.004, V["hold"][5]))
    gmeta["marks"] = []
    for kk, cx in enumerate(plan_cuts):
        st0 = int(np.flatnonzero(PH == f"cut{kk}")[0])
        _r = math.radians(args.knife_roll_deg)  # 팀원 knife._axes(yaw −90°, roll) 의 e_v = (cos r, 0, sin r)
        _ev = (math.cos(_r), 0.0, math.sin(_r)) if args.knife_roll_deg else None
        mk = dict(step=st0, phase=f"cut{kk}", x_mm=float(cx * 1000), y_mm=0.0, yaw_deg=0.0, z_mm=float(Qg[st0, 2] * 1000))
        if _ev is not None:
            mk["nrm"] = [float(v) for v in _ev]  # 칼질 평면 법선(우리 좌표) — 메쉬 나누기(render_surface.split_pieces)가 읽음
        gmeta["marks"].append(mk)
    gmeta["knife_roll_deg"] = args.knife_roll_deg
    gmeta["z_lift"] = 0.0
    gmeta["plan_mm"] = th
PLAN2 = None
if args.plan2:  # 11-19 reports/54: 편썰기 → (순간 이동) 조각 눕혀 쌓기 → 채썰기, 한 실행
    if not args.plan or args.knife_roll_deg:
        raise SystemExit("--plan2 는 --plan(편썰기) 뒤에만, 칼 기울이기 없이")
    th2 = [float(v) for v in args.plan2.split(",")]
    PH = np.where(PH == "rest", "rest1", PH)  # 편썰기 끝 멈춤(마지막 멈춤 'rest' 와 구별)
    k0 = len(plan_cuts)
    z_top2 = sum(th[:args.stack_n] if args.stack_n > 0 else th) / 1000.0 * 1.35 + 0.001  # 쌓은 높이 어림: 편 하나가 입자 층 때문에 3.6~4.4 mm(11-19 S_k010 으로 계산) → 두께 합 × 1.35 + 1 mm. 실제 높이는 옮길 때 로그에
    lift2 = max(args.lift, z_top2 + args.lift_clear2)
    cuts2 = [float(v) for v in args.stack_x - np.cumsum(th2) / 1000.0]
    q2, ph2 = [], []

    def seg2(name, a, b, ms):
        ms = max(1, int(round(ms)))
        for j in range(ms):
            q2.append(a + (b - a) * (j + 1) / ms)
            ph2.append(name)

    cur = Qg[-1].copy()
    seg2("relocate", cur, cur, 1)
    b = np.array([cuts2[0], 0.0, lift2, 0.0])
    seg2("settle2", cur, b, args.settle2_s * 1000)
    cur = b
    for kk, cx in enumerate(cuts2):
        kq = k0 + kk
        if kk > 0:
            b = np.array([cx, 0.0, lift2, 0.0])
            seg2(f"move{kq}", cur, b, abs(b[0] - cur[0]) / 0.08 * 1000)
            cur = b
        b = np.array([cx, 0.0, z_top2 + 0.0035, 0.0])
        seg2(f"approach{kq}", cur, b, abs(cur[2] - b[2]) / 0.05 * 1000)
        a, b = b, np.array([cx, 0.0, 0.0005, 0.0])
        seg2(f"cut{kq}", a, b, (a[2] - b[2]) / CUT_SPEED * 1000)  # 11-25: 기본 0.04 m/s(--cut_speed)
        if args.bottom_saw_s > 0:  # 11-21 저녁: 채썰기에도 편썰기와 같은 바닥 톱질(생 돼지 기본)
            seg2(f"bsaw{kq}", b, b, args.bottom_saw_s * 1000)
        seg2(f"dwell{kq}", b, b, 50)
        a, b = b, np.array([cx, 0.0, lift2, 0.0])
        seg2(f"withdraw{kq}", a, b, (b[2] - a[2]) / 0.05 * 1000)
        seg2(f"top{kq}", b, b, args.top_s * 1000)
        cur = b
    seg2("rest", cur, cur, args.rest2_s * 1000)
    n1 = len(Qg)
    Qg = np.concatenate([Qg, np.array(q2)])
    PH = np.concatenate([PH, np.array(ph2)])
    if args.saw_amp > 0:  # 11-21 저녁: 채썰기 칼질에도 밀며 썰기(편썰기와 같은 진폭·빈도)
        _add_saw(range(k0, k0 + len(cuts2)))
    for kk, cx in enumerate(cuts2):
        st0 = int(np.flatnonzero(PH == f"cut{k0 + kk}")[0])
        gmeta["marks"].append(dict(step=st0, phase=f"cut{k0 + kk}", x_mm=float(cx * 1000), y_mm=0.0, yaw_deg=0.0,
                                   z_mm=float(Qg[st0, 2] * 1000)))
    gmeta["plan2_mm"] = th2
    PLAN2 = dict(k0=k0, cuts=cuts2, z_top_plan=z_top2, x_end=args.stack_x, relocate_step=n1, th=th2)
    for _n, _c in gmeta.get("cams", {}).items():  # 몸통과 쌓은 편이 함께 보이게 카메라를 +x 로 2 cm, 단면 쪽은 1.5배 뒤로(채썰기 칼이 화면을 가리지 않게)
        _p, _l = np.array(_c["pos"], float), np.array(_c["lookat"], float)
        _l2 = _l + np.array([0.02, 0.0, 0.0])
        _p2 = _l2 + (_p - _l) * (1.5 if _n == "face" else 1.0)
        _c["pos"], _c["lookat"] = [float(v) for v in _p2], [float(v) for v in _l2]
if args.knife_floor is not None:
    Qg[:, 2] = np.maximum(Qg[:, 2], args.knife_floor)
if args.max_steps:
    Qg, PH = Qg[:args.max_steps], PH[:args.max_steps]
n_steps = len(Qg)
step_dt = float(gmeta.get("dt", 1e-3))

dt = 1.0 / args.physics_hz
if not args.no_og:
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("OMNIGIBSON_DATA_PATH", str(Path(__file__).resolve().parents[3] / "deps/BEHAVIOR-1K/datasets"))
    import omnigibson as og  # noqa: E402
    from omnigibson.macros import gm  # noqa: E402

    from simfoundry.utils.og_utils import apply_teleop_omnigibson_macros  # noqa: E402

    apply_teleop_omnigibson_macros(enable_tr=False)  # 팀원 tofu_og_cut.py 와 같은 설정(변환 규칙 끔)
    gm.ENABLE_OBJECT_STATES = False
    gm.ENABLE_TRANSITION_RULES = False
    og.launch(physics_dt=dt, rendering_dt=dt, sim_step_dt=dt, device="cpu")
# Isaac Sim 의 Warp 가 올라온 뒤에야 솔버와 보정 커널(한 프로세스에 Warp 하나)
import warp as wp  # noqa: E402

if args.wp_det:  # 11-23 reports/63 시험용: 솔버·보정 커널 모듈을 불러오기 전에 정해야 모든 커널에 걸림
    wp.config.deterministic = wp.DeterministicMode(int(args.wp_det))
    print(f"  Warp 결정적 모드: {wp.config.deterministic!r}", flush=True)

from simfoundry.cutting.cosim import matrix_to_quat_xyzw  # noqa: E402
from simfoundry.cutting.knife import KnifeGeometry, KnifePose, KnifeTrajectory  # noqa: E402
from simfoundry.cutting.materials import PRESETS  # noqa: E402
from simfoundry.cutting.mpm import MPMCuttingSim, SimConfig  # noqa: E402
from simfoundry.cutting.og_cosim import FoodPoints, write_knife_usd  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from warp_fixes import ADHESION_FILM, ADHESION_FILM2, ADHESION_FILM3, Fixes  # noqa: E402

if args.wp_nofma:  # 11-23 reports/63 시험용: 커널을 처음 띄우기 전에 정해야 그 모듈이 FMA 없이 컴파일됨
    for _m in ("simfoundry.cutting.mpm", "warp_fixes"):
        wp.set_module_options({"fuse_fp": False}, module=_m)
    print("  Warp FMA 묶기 끔(fuse_fp=False): simfoundry.cutting.mpm, warp_fixes", flush=True)

import dataclasses  # noqa: E402

L, R0, R1 = V["L"], V["r0"], V["r1"]
block = (X1 - X0, Y1 - Y0, ZT)
if V["material"] == "potato":
    mat = dataclasses.replace(PRESETS["cucumber"], name="potato", youngs_modulus=3.0e6, poisson_ratio=0.35, density=1090.0,
                              fracture_energy=300.0, tensile_strength=3.0e5)
elif V["material"] in NEW_MATS:  # 11-20 새 식재료
    _b, _kw = NEW_MATS[V["material"]]
    mat = dataclasses.replace(PRESETS[_b], name=V["material"], **_kw)
else:
    mat = PRESETS[V["material"]]
if args.E is not None:
    mat = dataclasses.replace(mat, youngs_modulus=args.E, tensile_strength=args.E / 10.0)
if args.Gc is not None:
    mat = dataclasses.replace(mat, fracture_energy=args.Gc)
if args.tensile is not None:  # 11-20 저녁 reports/58: 찢김(tearing) 문턱인 인장 강도만 바꾸는 시험
    mat = dataclasses.replace(mat, tensile_strength=args.tensile)
if args.relaxed_E is not None:  # 11-21 새벽 reports/58: --E 와 함께 이완 영률도(0 < relaxed < E 여야 이완이 켜짐)
    mat = dataclasses.replace(mat, relaxed_modulus=args.relaxed_E)
if args.rho is not None:  # 11-19 reports/53: 두께는 그대로 두고 무게만 바꾸는 시험
    mat = dataclasses.replace(mat, density=args.rho)
if args.knife_friction is not None:  # 11-17: 솔버 set_knife 가 작은 단계마다 material.knife_friction 을 칼 상태로 복사
    mat = dataclasses.replace(mat, knife_friction=args.knife_friction)
if args.contact_friction is not None:  # 11-17 저녁(reports/47): 조각끼리 마찰 덮어쓰기(진단)
    mat = dataclasses.replace(mat, contact_friction=args.contact_friction)
k = gmeta["knife"]
if args.knife_bevel is not None:  # 11-14: 칼 쐐기 높이 바꿈(렌더러도 meta 의 칼 치수를 씀)
    k["bevel_height"] = float(args.knife_bevel)
knife_geo = KnifeGeometry(half_length=k["length"] / 2, height=k["height"], edge_thickness=k["edge_width"],
                          spine_thickness=k["thickness"], bevel_height=k["bevel_height"], handle_length=0.11)
if args.top_margin is None:  # 접착을 켜면 칼에 붙은 조각이 칼등 근처까지 올라갈 수 있다: 칼 드는 높이 + 칼 높이 + 15 mm + 벽 띠(4.5칸)
    args.top_margin = (max(0.09, float(Qg[:, 2].max()) + float(gmeta["knife"]["height"]) + 0.015 - ZT + 4.5 * args.dx)
                       if args.adhesion != "none" else 0.035)
_cfg_kw = dict(max_pieces=args.max_pieces) if args.max_pieces else {}
if args.dilation is not None:
    _cfg_kw["dilation"] = float(args.dilation)
if args.cfl is not None:  # 11-23 reports/62 시간 간격 비교
    _cfg_kw["cfl"] = float(args.cfl)
if args.no_tearing:  # 11-20 저녁 reports/58: 덜 잘린 곳이 인장 강도에서 저절로 찢기는 규칙(SimConfig.tearing) 끔 → 균열은 G_c 셈으로만
    _cfg_kw["tearing"] = False
cfg = SimConfig(dx=args.dx, side_margin=args.side_margin, top_margin=args.top_margin, face_contact=args.face_contact, **_cfg_kw)


if SH:
    _occ = np.load(SH["occ"])
    OCC_LO, OCC_H, OCC_SHAPE = _occ["lo"], float(_occ["h"]), tuple(int(v) for v in _occ["shape"])
    OCC = np.unpackbits(_occ["occ"])[:int(np.prod(OCC_SHAPE))].reshape(OCC_SHAPE).astype(bool)


def inside(p):  # 상자 아래 모서리 기준
    if SH:  # 실제 모양: 점유 격자(0.25 mm 칸)
        q = p + np.array([X0, Y0, 0.0])
        ijk = np.floor((q - OCC_LO) / OCC_H).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(OCC_SHAPE)), axis=1)
        res = np.zeros(len(p), bool)
        res[ok] = OCC[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        return res
    # 원뿔대(당근)·원기둥: x 를 따라 반지름 R0 → R1, 바닥을 도마에 붙임
    r = R0 + (R1 - R0) * np.clip(p[:, 0] / L, 0.0, 1.0)
    return (p[:, 1] - R0) ** 2 + (p[:, 2] - r) ** 2 <= r * r


sim = MPMCuttingSim(mat, block, knife=knife_geo, config=cfg, device=args.device, inside=inside)
if FAST3_AUTO and not all(hasattr(sim, a) for a in ("fast_box", "fast_launch", "fast_py", "_update_box_pc", "fast_gmask")):
    # 11-24: 기본 빠른3 인데 불러온 솔버에 빠른 고침이 없음(원본만 PYTHONPATH 에 있거나 팀원 원본 솔버) → 원래 코드로(결과 같음, 느림)
    args.fast3 = False
    FAST3_NOTE = FAST3_NOTE or f"불러온 솔버({Path(sys.modules[MPMCuttingSim.__module__].__file__).parents[2]})에 빠른3 고침이 없음 → 원래 코드로"
    print(f"  [경고] 빠른3 기본 못 씀: {FAST3_NOTE}", flush=True)
if args.fast3:  # 11-23 reports/63
    args.fast2 = args.fast_gmask = True
if args.fast or args.fast2:  # 11-23 reports/62
    args.fast_box = args.fast_diag = args.fast_launch = args.fast_py = True
if args.fast2:  # 11-23 reports/63
    args.fast_pcbox = args.fast_fuse = args.fast_sel2 = True
if args.fast_box or args.fast_diag or args.fast_launch or args.fast_py:  # 11-23 reports/62
    if not hasattr(sim, "fast_box") or (args.fast_launch and not hasattr(sim, "fast_launch")):
        raise SystemExit("--fast_box/--fast_diag: fast-box 패치 복사본(<빠른3 솔버 복사본>)을 PYTHONPATH 에 둘 것")
    sim.fast_box, sim.box_margin, sim.fast_diag = int(args.fast_box), int(args.box_margin), int(args.fast_diag)
    if args.fast_launch:
        sim.fast_launch = 1
    if args.fast_py:
        if not hasattr(sim, "fast_py"):
            raise SystemExit("--fast_py: make_warp_fast_py.py 복사본이 필요")
        sim.fast_py = 1
if args.fast_pcbox or args.fast_fuse:  # 11-23 reports/63
    if not hasattr(sim, "_update_box_pc"):
        raise SystemExit("--fast_pcbox/--fast_fuse: fast2 패치 복사본(<빠른3 솔버 복사본>)을 PYTHONPATH 에 둘 것")
    if args.fast_pcbox:
        sim.fast_box, sim.box_margin = 2, int(args.box_margin)
    if args.fast_fuse:
        sim.fast_fuse = 1
if args.fast_poison:  # 11-23 reports/63 시험용
    if not hasattr(sim, "fast_poison"):
        raise SystemExit("--fast_poison: 독약 시험 복사본(<빠른3 솔버 복사본>)을 PYTHONPATH 에 둘 것")
    sim.fast_poison = 1
if args.fast_gmask:  # 11-23 reports/63
    if not hasattr(sim, "fast_gmask"):
        raise SystemExit("--fast_gmask: fast3 패치 복사본(<빠른3 솔버 복사본>)을 PYTHONPATH 에 둘 것")
    sim.fast_gmask = 1
off = sim.block_min - np.array([X0, Y0, 0.0])  # 우리 좌표 + off = 솔버 좌표
X_REST0 = (np.asarray(sim.x0, float) - off).copy()  # 11-19: 이어 하기에서 옮긴 입자의 처음 자리를 바꿔도 렌더는 원래 처음 자리로
if args.planks:  # 채썰기 시연: 편을 처음부터 따로 된 조각으로(편 경계 y 로 나눔, 서로 붙지 않고 맞닿기만)
    _xo = sim.x0 - off
    _pid = np.digitize(_xo[:, int(SH.get("plank_axis", 1))], SH["plank_edges"]).astype(np.int32)  # 11-14: 쌓은 편(--stack)은 z
    sim.piece = wp.array(_pid, dtype=int, device=args.device)
    sim.n_pieces = int(_pid.max()) + 1
    print(f"  편 {sim.n_pieces}장(입자 {np.bincount(_pid).tolist()})", flush=True)
elif SH and SH.get("layers") and not args.no_layers:  # 11-20: 양파 비늘·대파 잎집 = 처음부터 따로 된 겹(서로 붙지 않고 맞닿기만)
    _xo, LY = sim.x0 - off, SH["layers"]
    if LY["kind"] == "ellipsoid":
        _rho = np.linalg.norm((_xo - np.array(LY["center"])) / np.array(LY["radii"]), axis=1)
    else:  # 원기둥: 축(axis)에서 잰 반지름
        _o = [k for k in range(3) if k != int(LY["axis"])]
        _rho = np.hypot(_xo[:, _o[0]] - LY["center"][0], _xo[:, _o[1]] - LY["center"][1])
    _pid = np.digitize(_rho, LY["edges"]).astype(np.int32)  # 0 = 가운데, 큰 번호 = 바깥 겹
    sim.piece = wp.array(_pid, dtype=int, device=args.device)
    sim.n_pieces = int(_pid.max()) + 1
    print(f"  겹 {sim.n_pieces}개(입자 {np.bincount(_pid).tolist()})", flush=True)
if SH and SH.get("shell") and args.shell_E:  # 11-20: 껍질 막(소시지) — 겉 두께 안 입자만 단단하게
    if not hasattr(sim, "mscale"):
        raise SystemExit("--shell_E 는 patches/make_warp_mod_scale.py 를 적용한 복사본(<빠른3 솔버 복사본>)이 필요")
    _xo, SHL = sim.x0 - off, SH["shell"]
    _ax = int(SHL["axis"])
    _o = [k for k in range(3) if k != _ax]
    _xc = np.clip(_xo[:, _ax], -SHL["half_len"], SHL["half_len"])
    _d = SHL["radius"] - np.sqrt((_xo[:, _ax] - _xc) ** 2 + (_xo[:, _o[0]] - SHL["center"][0]) ** 2
                                 + (_xo[:, _o[1]] - SHL["center"][1]) ** 2)
    _sc = float(args.shell_E) / mat.youngs_modulus
    _ms = np.where(_d < SHL["thickness"], _sc, 1.0).astype(np.float32)
    sim.mscale = wp.array(_ms, dtype=float, device=args.device)
    sim.dt = sim.dt / math.sqrt(max(_sc, 1.0))
    print(f"  껍질 막: 입자 {int((_ms > 1).sum())}개(두께 {SHL['thickness'] * 1e3:.1f} mm) 탄성 ×{_sc:.0f}, 시간 간격 {sim.dt * 1e6:.2f} µs",
          flush=True)
    if args.shell_gc_mul != 1.0:  # 11-20 reports/57: 껍질 띠 G_c 배율(질김)
        if not hasattr(sim, "shell_mul"):
            raise SystemExit("--shell_gc_mul 은 patches/make_warp_shell_toughness.py 를 적용한 복사본이 필요")
        sim.shell_t, sim.shell_mul = float(SHL["thickness"]), float(args.shell_gc_mul)
        if args.shell_gc_mode == "particle":  # 11-20 저녁 reports/58: 칸마다 껍질 입자 구간(옆면·아래 껍질 포함)
            if not hasattr(sim, "shell_mode"):
                raise SystemExit("--shell_gc_mode particle 은 patches/make_warp_shell_toughness2.py 를 적용한 복사본(<빠른3 솔버 복사본>)이 필요")
            sim.shell_mode = 1
        print(f"  껍질 질김({args.shell_gc_mode}): " + (f"윗면 {sim.shell_t * 1e3:.1f} mm 띠" if args.shell_gc_mode == "band" else "칸마다 껍질 입자 구간")
              + f" G_c ×{sim.shell_mul:g} = {mat.fracture_energy * sim.shell_mul:.0f} J/m²", flush=True)
if args.region_E_mul is not None:  # 11-20 저녁 reports/58: 처음 자리 x 가 큰 쪽(대파 초록 잎)만 영률 배율
    if not hasattr(sim, "mscale"):
        raise SystemExit("--region_E_mul 은 patches/make_warp_mod_scale.py 를 적용한 복사본(<빠른3 솔버 복사본>·7)이 필요")
    _ms = sim.mscale.numpy().astype(np.float32)
    _sel = (sim.x0 - off)[:, 0] > args.region_xmin
    _ms[_sel] *= float(args.region_E_mul)
    sim.mscale = wp.array(_ms, dtype=float, device=args.device)
    if args.region_E_mul > 1.0:
        sim.dt = sim.dt / math.sqrt(float(args.region_E_mul))
    print(f"  부분 영률: x > {args.region_xmin * 1e3:.1f} mm 입자 {int(_sel.sum())}개 영률 ×{args.region_E_mul:g}", flush=True)
if args.blade_split:
    if not hasattr(sim, "blade_split"):
        raise SystemExit("--blade_split 은 patches/make_warp_blade_split.py 를 적용한 복사본이 필요")
    sim.blade_split = 1
if args.knife_side:
    if not hasattr(sim, "knife_side"):
        raise SystemExit("--knife_side 는 patches/make_warp_knife_side.py 를 적용한 복사본이 필요")
    sim.knife_side = 1
if args.gap_contact > 0:
    if not hasattr(sim, "enable_gap_contact"):
        raise SystemExit("--gap_contact 는 patches/make_warp_gap_contact.py 를 적용한 복사본이 필요")
    sim.enable_gap_contact(args.gap_contact)
if args.reside:
    if not hasattr(sim, "reside"):
        raise SystemExit("--reside 는 patches/make_warp_reside.py 를 적용한 복사본이 필요")
    sim.reside = 1
if args.pinch_extra:  # 11-14 밤: 바닥 가르기 문턱 넓히기(복사본 patches/make_warp_pinch.py)
    if "pinch" not in type(sim._knife).__dict__.get("__annotations__", {}) and not hasattr(sim._knife, "pinch"):
        raise SystemExit("솔버 복사본에 pinch 패치가 없음: python patches/make_warp_pinch.py")
    sim.pinch_extra = float(args.pinch_extra)
if args.face_tan:
    if not hasattr(sim, "face_tan") or args.gap_contact <= 0:
        raise SystemExit("--face_tan 은 --gap_contact 와 patches/make_warp_face_tangent.py 를 적용한 복사본이 필요")
    sim.face_tan = 1
if args.kerf_slip:  # 11-17 저녁(reports/47)
    if not hasattr(sim, "kerf_slip"):
        raise SystemExit("--kerf_slip 은 patches/make_warp_kerf_slip.py 를 적용한 복사본이 필요")
    sim.kerf_slip = 1
    sim.kerf_depth_cells = float(args.kerf_depth)
fixes = None
if args.pull_check and args.no_fixes:
    raise SystemExit('--pull_check 는 손(Fixes)이 있어야 함(--no_fixes 와 같이 못 씀)')
if not args.no_fixes:
    if not hasattr(sim, "board_minz"):
        raise SystemExit("고친 판은 도마 접촉 고친 복사본이 필요: PYTHONPATH=<빠른3 솔버 복사본>:<이 저장소>")
    adh = None
    if args.adhesion in ("film", "film2", "film3"):
        adh = dict({"film": ADHESION_FILM, "film2": ADHESION_FILM2, "film3": ADHESION_FILM3}[args.adhesion])
        for _k in adh:
            if getattr(args, f"adh_{_k}") is not None:
                adh[_k] = getattr(args, f"adh_{_k}")
        for _k in ("p_a", "c_n", "c_t", "tau", "sigma"):  # 수분(물막) 세기 배율
            adh[_k] *= args.wet
        if args.adh_layers:
            adh["layers"] = args.adh_layers
        if args.adh_anchor:  # 11-17 reports/47: 조각 면 맞춤
            adh.update(anchor=1, anchor_q=0.1, anchor_lo=-0.3e-3, anchor_hi=args.adh_anchor_hi)
        adh = {k_: v_ for k_, v_ in adh.items() if k_ != "layers" or v_}
    fixes = Fixes(sim, off, ZT + float(gmeta.get("z_lift", 0.0)), device=args.device, hold_box=V["hold"], step_dt=step_dt,
                  adhesion=adh, adh_release=args.adh_release, bounce_damp=args.bounce_damp, release_outer=args.release_outer,
                  hold_mode=args.hold_mode, hold_k=args.hold_k, hold_fmax=args.hold_fmax, hold_tau=args.hold_tau, hold_zclamp=args.hold_zclamp,
                  end_damp=args.end_damp, end_h=args.end_h, end_scope=args.end_scope,
                  spine_topple=args.spine_topple, spine_near=args.spine_near, spine_cool=args.spine_cool, spine_max=args.spine_max,
                  spine_below=args.spine_below)
    fixes.fast_select = 2 if args.fast_sel2 else bool(args.fast_py)  # 11-23 reports/62·63: 감쇠 조각 고르기 빠른 경로(결과 같음)
pair_info = None
if args.pair_adh:  # 11-18 reports/48: 조각끼리 접착 — 칼 접착(film × wet)과 같은 물막 값
    if not hasattr(sim, "pair_adh"):
        raise SystemExit("--pair_adh 는 patches/make_warp_pair_adhesion.py 를 적용한 복사본이 필요")
    if not args.gap_contact:
        raise SystemExit("--pair_adh 는 --gap_contact(조각끼리 닿음 판정)가 필요")
    _a = (fixes.adhesion if (fixes is not None and fixes.adhesion is not None) else
          {k_: v_ * args.wet for k_, v_ in ADHESION_FILM3.items() if k_ in ("p_a", "c_n", "tau", "c_t")})
    sim.pair_adh = 1
    sim.adh_pa, sim.adh_cn = float(_a["p_a"]) * args.pair_adh_scale, float(_a["c_n"]) * args.pair_adh_scale
    sim.adh_tau, sim.adh_ct = float(_a["tau"]) * args.pair_adh_scale, float(_a["c_t"]) * args.pair_adh_scale
    sim.adh_gap = float(args.pair_adh_gap if args.pair_adh_gap is not None else args.gap_contact)
    sim.adh_face_cos = float(args.pair_adh_face_cos)
    pair_info = dict(pa=sim.adh_pa, cn=sim.adh_cn, tau=sim.adh_tau, ct=sim.adh_ct, gap_cells=sim.adh_gap, face_cos=sim.adh_face_cos,
                     scale=args.pair_adh_scale, body=bool(args.pair_adh_body))
    print(f"  조각끼리 접착: {pair_info}", flush=True)
spacing = args.dx / cfg.particles_per_axis

# 장면: 탁자(높이 0.76 m) 위 도마(두께 18 mm), 그 위 채소. 우리 좌표 → 월드: + (0, 0, ZB)
TABLE_H, BOARD_T = 0.76, 0.018
ZB = TABLE_H + BOARD_T
W_OFF = np.array([0.0, 0.0, ZB])
knife_usd = None if args.no_og else write_knife_usd(out / "knife" / "knife.usd", knife_geo)


ROLL = math.radians(args.knife_roll_deg)


def knife_pose_our(q):  # 손잡이가 카메라 반대쪽(-y 쪽 카메라 기준)으로 가도록 회전 −90° (칼날은 양쪽이 같아 물리는 같음)
    return KnifePose(origin=np.array(q[:3]), yaw=float(q[3]) - math.pi / 2, roll=ROLL)


def knife_quat(p):  # OmniGibson 칼 방향: 칼 축(e_u, e_v, e_w)을 열로 한 회전(roll 0 이면 z 축 yaw 회전과 같음)
    return [float(v) for v in matrix_to_quat_xyzw(np.stack(p.axes(), axis=1))]


def solver_pose(q):
    p = knife_pose_our(q)
    return KnifePose(origin=p.origin + off, yaw=p.yaw, roll=p.roll)


def look_at(eye, target):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return matrix_to_quat_xyzw(np.stack([right, up, -f], axis=1))


hz = args.physics_hz
writers, cams = {}, {}
import imageio.v2 as imageio  # noqa: E402
if not args.no_og:
    sensors = []
    for cam in ("persp", "side", "face"):
        c = gmeta["cams"][cam]
        eye, tgt = np.array(c["pos"]) + W_OFF, np.array(c["lookat"]) + W_OFF
        sensors.append({"sensor_type": "VisionSensor", "name": cam, "modalities": ["rgb"],
                        "sensor_kwargs": {"image_height": args.height, "image_width": args.width, "clipping_range": [0.01, 20.0]},
                        "position": [float(v) for v in eye], "orientation": [float(v) for v in look_at(eye, tgt)]})
    objects = [
        dict(type="PrimitiveObject", name="table", primitive_type="Cube", fixed_base=True, visual_only=True,
             rgba=[0.86, 0.85, 0.82, 1.0], scale=[0.9, 0.7, TABLE_H], position=[0.0, 0.0, TABLE_H / 2]),
        dict(type="PrimitiveObject", name="board", primitive_type="Cube", fixed_base=True, visual_only=True,
             rgba=[0.50, 0.33, 0.18, 1.0], scale=[0.34, 0.24, BOARD_T], position=[0.0, 0.0, TABLE_H + BOARD_T / 2]),
        dict(type="USDObject", name="knife", usd_path=str(knife_usd), category="knife", visual_only=True, fixed_base=True,
             position=[0.0, 0.0, ZB + 0.06]),
    ]
    hz = args.physics_hz
    env = og.Environment(configs=dict(
        env=dict(action_frequency=hz, rendering_frequency=hz, physics_frequency=hz, external_sensors=sensors),
        scene=dict(type="Scene", use_floor_plane=True, floor_plane_visible=True, use_skybox=True),
        objects=objects))
    env.reset()
    cams = {s["name"]: env.external_sensors[s["name"]] for s in sensors}
    knife_obj = env.scene.object_registry("name", "knife")

    # 채소 입자(그리기): 겉껍질 입자 색, 속살은 바깥 → 가운데 색
    x0o = sim.x0 - off
    rr = np.sqrt(x0o[:, 1] ** 2 + (x0o[:, 2] - R0) ** 2) / max(R0, 1e-9)
    fi, fo = np.array(V["flesh_in"]), np.array(V["flesh_out"])
    albedo = np.where(sim.skin[:, None], np.array(V["skin"]), fi[None] * (1 - rr[:, None]) + fo[None] * rr[:, None])
    points = FoodPoints(og.sim.stage, "/World/mpm_food", sim.skin, 1.6 * spacing, None, None, albedo=albedo)


    def to_world(x_solver):
        return x_solver - off + W_OFF



    # Isaac Sim 화면은 USD 변경을 한두 번 그린 뒤에야 보여 준다 → 첫 장면 전에 몇 번 그려 둔다(안 하면 첫 프레임에 채소·칼이 없음)
    _p0 = knife_pose_our(Qg[0])
    knife_obj.set_position_orientation(position=(_p0.origin + W_OFF).tolist(), orientation=knife_quat(_p0))
    points.update(to_world(sim.positions()))
    for _ in range(4):
        og.sim.render()

    writers = {c: imageio.get_writer(str(out / f"{c}.mp4"), fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
               for c in cams}

if args.saw_sever:  # 11-21 reports/59: 도마 위 질긴 아래층은 톱질해야 끊김
    if not hasattr(sim, "saw_h"):
        raise SystemExit("--saw_sever 는 patches/make_warp_saw_sever.py 를 적용한 복사본(<빠른3 솔버 복사본>)이 필요")
    sim.saw_h, sim.saw_len = (float(v) / 1000.0 for v in args.saw_sever.split(","))
    if sim.saw_h <= sim.dx / sim.config.particles_per_axis:
        raise SystemExit(f"--saw_sever 층 두께는 입자 간격 {sim.dx / sim.config.particles_per_axis * 1e3:.2f} mm 보다 커야 함(cut_complete)")
    if args.saw_region != "all":  # 11-21 reports/60: 일부 부위에만 — 입자마다 쉬는 자리(모양 좌표)로 질긴 조직 표시
        if not hasattr(sim, "saw_need"):
            raise SystemExit("--saw_region 은 make_warp_saw_sever.py 2판(11-21 오후)을 적용한 overlay8 이 필요")
        if not SH or not np.allclose(np.array(SH["xform"]), np.eye(4)):
            raise SystemExit("--saw_region 은 make_food_shapes.py 모양(xform 단위 행렬)에서만")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from food_regions import REGIONS  # noqa: E402
        sim.saw_tag = REGIONS[args.saw_region](X_REST0)  # 처음 자리(솔버 좌표 − off = 우리 좌표 = 모양 좌표)
        np.save(out / "saw_tag.npy", sim.saw_tag)
    print(f"  톱질 아래층: 도마 위 {sim.saw_h * 1e3:.2f} mm, 날이 그 안에서 {sim.saw_len * 1e3:.0f} mm 미끄러져야 끊김, "
          f"자리 {args.saw_region}" + (f"(질긴 조직 입자 {int(sim.saw_tag.sum())}/{len(sim.saw_tag)})" if getattr(sim, "saw_tag", None) is not None else "")
          + f", 바닥 톱질 {args.bottom_saw_s:g} s", flush=True)
elif V.get("material") == "pork":
    print("  톱질 아래층 끔(--saw_sever off) — 생 돼지 기본과 다름", flush=True)
frame_every = 1.0 / args.fps
next_frame = 0.0
frames = {"x": [], "lab": [], "step": [], "q": [], "lig": [], "t": []}  # t: 솔버 시간 [단계](11-24, 힘 한계 판은 step = 경로 줄과 다름)
dense = {"x": [], "v": [], "lab": [], "step": [], "q": [], "f": [], "front": [], "area_path": [], "cut": []}
snaps = {}
cut_steps = {i for i in range(n_steps) if str(PH[i]).startswith("cut") and (i == 0 or PH[i - 1] != PH[i])}


def hand_box(xmax):  # 손 상자(우리 좌표): x 는 채소 왼쪽 끝 ~ xmax, y 는 채소 폭 + 4 mm
    return ((X0 + xmax) / 2, (Y0 + Y1) / 2, V["hold"][2], xmax - X0, (Y1 - Y0) + 0.004, V["hold"][5])


# 11-19 reports/54 이어 하기(--plan2): 편썰기 조각을 눕혀 쌓기
R_LAY = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # 처음 자리 기준 눕힘: x(편 두께) → z, z → −x (stack12 와 같은 돌림)
RELOC = {}


def _kabsch(A, B):  # B ≈ R A + t
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def _R2q(R):
    t = np.trace(R)
    if t > 0:
        s_ = np.sqrt(t + 1.0) * 2
        q = [0.25 * s_, (R[2, 1] - R[1, 2]) / s_, (R[0, 2] - R[2, 0]) / s_, (R[1, 0] - R[0, 1]) / s_]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s_ = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s_, 0.25 * s_, (R[0, 1] + R[1, 0]) / s_, (R[0, 2] + R[2, 0]) / s_]
    elif R[1, 1] > R[2, 2]:
        s_ = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s_, (R[0, 1] + R[1, 0]) / s_, 0.25 * s_, (R[1, 2] + R[2, 1]) / s_]
    else:
        s_ = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s_, (R[0, 2] + R[2, 0]) / s_, (R[1, 2] + R[2, 1]) / s_, 0.25 * s_]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _slerp_R(R, u):  # 단위 회전에서 R 까지 u(0~1)만큼
    q = _R2q(R)
    if q[0] < 0:
        q = -q
    ang = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    if ang < 1e-8:
        return np.eye(3)
    ax = q[1:] / np.sin(ang / 2)
    a = ang * u
    w, x, y, z = np.cos(a / 2), *(ax * np.sin(a / 2))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def hand_box2(xmax):  # 채썰기 손 상자: 쌓은 편 왼쪽 끝(+x 끝 − 채소 높이 − 3 mm) ~ xmax
    x0s = PLAN2["x_end"] - ZT - 0.003
    return ((x0s + xmax) / 2, (Y0 + Y1) / 2, V["hold"][2], xmax - x0s, (Y1 - Y0) + 0.004, V["hold"][5])


def relocate(i):
    """편썰기 조각(몸통 뺀 200 입자 이상)을 눕혀(처음 자리 기준 R_LAY) 쌓는다: 마지막 편이 맨 아래, 첫 편이 맨 위(stack12 와 같은 순서),
    +x 끝을 stack_x 에 맞추고 y 가운데, 편 사이는 입자 한 간격 + stack_gap. 물리 = 순간 이동(속도·C 0, F = I, 처음 자리 = 새 자리).
    영상에만 옮기는 장면(조각 하나씩 들어 돌리며 옮겨 내려놓음)을 frames 에 끼워 넣는다(같은 step)."""
    X = sim.x.numpy().astype(float)
    lab = sim.pieces()
    cnt = np.bincount(lab)
    body = int(np.argmax(cnt))
    sl = [int(p) for p in np.flatnonzero(cnt) if p != body and cnt[p] >= 200]
    kept = []
    if args.stack_n > 0:  # 11-26 reports/79: 먼저 썬 편(조각 번호가 작은 쪽 = 칼질 차례)만 쌓음
        sl, kept = sorted(sl)[:args.stack_n], sorted(sl)[args.stack_n:]
    xo = X - off
    new = xo.copy()
    z_next = 0.5 * spacing
    moves = []
    base_of = {}  # 11-26: --stack_from_rest 옮기는 장면용 — 처음 모양을 지금 방향으로 돌린 것(늘어난 편도 깨끗한 편으로 옮겨 보이게)
    for p in sorted(sl, reverse=True):
        m = lab == p
        Rk, _ = _kabsch(X_REST0[m], xo[m])
        Rm = R_LAY @ Rk.T
        ca = xo[m].mean(0)
        Pm = (xo[m] - ca) @ Rm.T
        ext_now = np.ptp(xo[m], axis=0)
        if args.stack_from_rest:  # 11-26: 처음 모양을 눕혀 놓음(= 강체 움직임이면 위 Pm 과 같음)
            Pm = (X_REST0[m] - X_REST0[m].mean(0)) @ R_LAY.T
            base_of[p] = (X_REST0[m] - X_REST0[m].mean(0)) @ Rk.T
        print(f"  쌓을 조각 {p}: 지금 크기 {np.round(1e3 * ext_now, 1).tolist()} mm, 처음 크기 {np.round(1e3 * np.ptp(X_REST0[m], axis=0), 1).tolist()} mm, "
              f"가운데 {np.round(1e3 * ca, 1).tolist()} mm", flush=True)
        sh = np.array([PLAN2["x_end"] - 0.5 * spacing - Pm[:, 0].max(), -Pm[:, 1].mean(), z_next - Pm[:, 2].min()])
        new[m] = Pm + sh
        z_next = float(new[m][:, 2].max()) + spacing + args.stack_gap
        moves.append((p, m, Rm, ca))
    if args.carry_frames > 0:
        cur = xo.copy()
        for p, m, Rm, ca in moves:
            cb = new[m].mean(0)
            # 1) 칼에 붙었거나 칼 옆에 선 조각은 칼에서 떼고(+x 22 mm·위 8 mm), 나머지는 들어 올림(20 mm) — 돌리지 않음
            # 2) 쌓을 자리 위(높이 = 내려놓을 자리 + 3 cm 이상)로 옮기며 돌림  3) 내려놓음. 11-19 시험에서 칼 옆에서 바로 돌려 칼을 뚫고 지나가던 것 고침
            near = abs(ca[0] - Qg[i][0]) < 0.02 and ca[2] > 0.004
            p1 = ca + (np.array([0.022, 0.0, 0.008]) if near else np.array([0.0, 0.0, 0.02]))
            p2 = np.array([cb[0], cb[1], max(p1[2], cb[2] + 0.03)])
            for f_ in range(args.carry_frames):
                u = (f_ + 1) / args.carry_frames
                if u <= 0.25:
                    w_ = u / 0.25
                    c, Ru = ca + (p1 - ca) * (w_ * w_ * (3 - 2 * w_)), np.eye(3)
                elif u <= 0.75:
                    w_ = (u - 0.25) / 0.5
                    w_ = w_ * w_ * (3 - 2 * w_)
                    c, Ru = p1 + (p2 - p1) * w_, _slerp_R(Rm, w_)
                else:
                    w_ = (u - 0.75) / 0.25
                    c, Ru = p2 + (cb - p2) * (w_ * w_ * (3 - 2 * w_)), Rm
                cur[m] = (base_of[p] if p in base_of else (xo[m] - ca)) @ Ru.T + c
                frames["x"].append(cur.astype(np.float32))
                frames["lab"].append(lab.astype(np.int8))
                frames["step"].append(i)
                frames["q"].append(Qg[i])
            cur[m] = new[m]
    mall = np.zeros(len(X), bool)
    for _, m, _, _ in moves:
        mall |= m
    Xs = X.copy()
    Xs[mall] = new[mall] + off
    sim.x.assign(Xs.astype(np.float32))
    if getattr(sim, "fast_box", 0):  # 11-23: 순간 이동은 단계 사이 일이라 상자 벗어남 검사에서 뺌
        sim._box_prev = None
    for arr in (sim.v, sim.C):
        a_ = arr.numpy()
        a_[mall] = 0.0
        arr.assign(a_)
    for arr in (sim.F, sim.Fn):
        a_ = arr.numpy()
        a_[mall] = np.eye(3, dtype=np.float32)
        arr.assign(a_)
    sim.x0[mall] = Xs[mall]
    sim.x0_dev.assign(np.asarray(sim.x0, np.float32))
    ztop = float(new[mall][:, 2].max()) + 0.5 * spacing
    if ztop > PLAN2["z_top_plan"] + 0.003:
        print(f"  !! 쌓은 높이 {ztop * 1000:.1f} mm 가 칼 경로 어림 {PLAN2['z_top_plan'] * 1000:.1f} mm + 3 mm 보다 높음 — 칼이 다가가는 중에 닿음", flush=True)
    if fixes is not None:
        fixes.z_top = ztop
    RELOC.update(step=i, pieces=[int(p) for p, *_ in moves], n=[int(m.sum()) for _, m, _, _ in moves], z_top_mm=round(ztop * 1000, 2),
                 x_mm=[round(float(new[mall][:, 0].min()) * 1000, 2), round(float(new[mall][:, 0].max()) * 1000, 2)],
                 body=body, left=[int(p) for p in np.flatnonzero(cnt) if p != body and 0 < cnt[p] < 200], kept=kept)
    print(f"  옮김(step {i}): 조각 {RELOC['pieces']}(입자 {RELOC['n']}) → 쌓은 높이 {RELOC['z_top_mm']} mm, x {RELOC['x_mm']} mm, "
          f"작은 조각 그대로 {RELOC['left']}, 안 쌓은 편 {RELOC['kept']}, 영상 프레임 {args.carry_frames * len(moves)}개 끼움", flush=True)


follow2, settle2_end = {}, -1
if PLAN2:  # --max_steps 로 잘린 경로에서도 되게(없는 단계는 건너뜀)
    for kk in range(1, len(PLAN2["cuts"])):
        _ix = np.flatnonzero(PH == f"approach{PLAN2['k0'] + kk}")
        if len(_ix):
            follow2[int(_ix[0])] = PLAN2["cuts"][kk]
    settle2_end = max((i_ for i_ in range(len(PH)) if PH[i_] == "settle2"), default=-1)


follow_steps = {}
if args.plan:
    for kk in range(1, len(plan_cuts)):
        _ix = np.flatnonzero(PH == f"approach{kk}")
        if len(_ix):  # 11-21 저녁: --max_steps 로 잘린 경로에서도 되게
            follow_steps[int(_ix[0])] = plan_cuts[kk]
settle_end = max((i for i in range(n_steps) if PH[i] == "settle"), default=0)
stopped_early = False
forces = np.zeros((n_steps, 3))
adh_log = np.zeros((n_steps, 2))  # [접착 걸린 입자·단계 수, 입자에 건 위쪽 힘 합(N·단계)]
adh_shift_log = None  # --adh_anchor: 단계마다 조각·쪽별 구역 이동 [m] (조각 수 × 2)
if fixes is not None and getattr(fixes, "adh_anchor", 0):
    adh_shift_log = np.zeros((n_steps, fixes.adh_shift.shape[0]), np.float32)
kf_cut = float(sim.material.knife_friction)  # 칼질(cut) 단계 칼 마찰(재료 프리셋 또는 --knife_friction)
gdiag_log = np.zeros((n_steps, 80, 3), np.float32) if hasattr(sim, "gdiag") else None  # 진단 복사본(reports/47 diag_overlay.py)만
gcnt_log = np.zeros((n_steps, 80), np.float32) if hasattr(sim, "gdiag") else None
pair_log = np.zeros((n_steps, 2), np.float32) if pair_info else None  # [접착 걸린 격자점·짝 수(부분단계 평균), 평균 접착 힘 크기 합 N]
pair_np = -1
render_s = mpm_s = 0.0
n_frames = 0
if args.pos_jitter > 0 and args.jitter_step <= 0:  # 11-23 reports/63 시험용: 입자 자리만 아주 조금 흔듦(처음 자리 x0·렌더 기준은 그대로)
    _xj = sim.x.numpy()
    _xj += np.random.default_rng(args.jitter_seed).normal(0.0, args.pos_jitter, _xj.shape).astype(np.float32)
    sim.x.assign(_xj)
    print(f"  입자 자리 흔들기: 표준편차 {args.pos_jitter:g} m, 씨앗 {args.jitter_seed}")

# ==== 실행기 본체: 여기부터 og_veg_cut.py 의 칼 경로 실행(food_robot_episodes.py 는 이 줄 앞 준비 부분만 이 프로세스에서 돌림) ====
# 11-24 reports/65: 칼 힘 한계(--force_cap, scripts/cutting/food/force_cap.py). 경로 줄 번호 i 와 솔버 시간 t 를 나눈다.
#   칼질(cut) 단계에서 내려가는 1 ms 마다 ForceGate.rate()(0~1)만큼만 내려가고, 덜 내려가면(adv False) 같은 줄 i 에 머문 채 시간 t 만 간다.
#   밀며 썰기(y 사인)는 시간에 따라 계속 — 늦어진 단계 수 H[k] 만큼 사인 위상을 민다. 멈춤(ForceGate.note)이면 그 칼질 남은 줄은
#   멈춘 높이 아래로 안 내려간다(칼을 그대로 들고 다음 칼질로). 한계가 없으면(GATE None) 아래 코드 길은 원래와 같다(t == i, row_first 늘 True).
GATE = None
if FORCE_CAP_N is not None:
    import re as _re
    GATE = ForceGate(FORCE_CAP_N, mode=args.force_mode, band=args.force_band,
                     filter_steps=max(1, int(round(args.force_filter_ms * 1e-3 / step_dt))), stall_s=args.stall_s,
                     stall_dz=args.stall_dz_mm * 1e-3, stall_max_s=args.stall_max_s, step_dt=step_dt)
    CUT_OF = np.full(n_steps, -1, np.int64)  # 줄 → 칼질 번호(cut·bsaw·dwell·withdraw 단계), 아니면 -1
    for _j in range(n_steps):
        _m = _re.match(r"(cut|bsaw|dwell|withdraw)(\d+)$", str(PH[_j]))
        if _m:
            CUT_OF[_j] = int(_m.group(2))
    SIG = np.full(n_steps, -1, np.int64)  # 줄이 그 칼질 cut·bsaw 단계의 몇째 줄인가(밀며 썰기 사인 번호), 아니면 -1
    DWM = np.full(n_steps, -1, np.int64)  # dwell 단계의 몇째 줄인가
    for _k, _tab in SAW_TAB.items():
        for _jj, _r in enumerate(_tab["ic"]):
            if _r < n_steps:
                SIG[_r] = _jj
        for _mm, _r in enumerate(_tab["iw"]):
            if _r < n_steps:
                DWM[_r] = _mm
    CAPST = dict(P=np.array(Qg[0], float), H={}, stalled={}, z_stall={}, cuts={}, r=1.0,
                 log=dict(t=[], i=[], q=[], f=[], r=[], adv=[], ff=[]))
    print(f"  힘 한계: {GATE.cap:g} N ({args.force_mode}, band {args.force_band:g}, 힘 {GATE.filter_steps} ms 평균, "
          f"멈춤 {args.stall_s:g} s/{args.stall_dz_mm:g} mm · 늦어짐 {args.stall_max_s:g} s)", flush=True)

    def _saw_val(k, j):  # 칼질 k 의 밀며 썰기 j 째 값(_add_saw 와 같은 식, 미리 만든 값이 있으면 그대로)
        tab = SAW_TAB[k]
        if j < len(tab["yc"]):
            return float(tab["yc"][j])
        return float(args.saw_amp * np.sin(2 * math.pi * args.saw_hz * (j + 1) * 1e-3))

    def _row_pose(j):  # 경로 줄 j 의 칼 자세: 늦어진 만큼 밀며 썰기 위상을 밀고, 멈춘 칼질은 멈춘 높이 아래로 안 내려감
        j = min(j, n_steps - 1)
        q = Qg[j]
        k = int(CUT_OF[j])
        if k < 0:
            return q
        H = CAPST["H"].get(k, 0)
        st_ = CAPST["stalled"].get(k, False)
        if H == 0 and not st_:
            return q
        q = np.array(q, float)
        if H > 0 and k in SAW_TAB:
            tab = SAW_TAB[k]
            if SIG[j] >= 0:
                q[1] = tab["base_ic"][SIG[j]] + _saw_val(k, int(SIG[j]) + H)
            elif DWM[j] >= 0:
                q[1] = tab["base_iw"][DWM[j]] + _saw_val(k, len(tab["ic"]) - 1 + H) * (1 - (DWM[j] + 1) / len(tab["iw"]))
        if st_:
            q[2] = max(q[2], CAPST["z_stall"][k])
        return q

    def cap_pose(i, t, first):
        """이번 1 ms 의 칼 시작·끝 자세(우리 좌표)와 '경로 줄 i 를 끝내나'(adv)."""
        a = CAPST["P"]
        b = _row_pose(i + 1)
        k = int(CUT_OF[i])
        CAPST["r"] = 1.0
        if first and i in cut_steps:  # 칼질 시작
            GATE.start_cut(k, a[2], t)
            CAPST["cuts"][k] = dict(k=k, t0=t, i0=i, z0=float(a[2]), z_min=float(a[2]), outcome=None, slow=0, fz_max=0.0,
                                    fz_sum=0.0, n_cut=0, fy_absmax=0.0)
        if not str(PH[i]).startswith("cut") or CAPST["stalled"].get(k, False) or i + 1 >= n_steps or b[2] >= a[2] - 1e-12:
            return a, b, True
        r = GATE.rate()
        CAPST["r"] = r
        if r >= 1.0:
            return a, b, True
        z_new = a[2] - (Qg[i, 2] - Qg[i + 1, 2]) * r  # 계획 내려가기(1 ms) × 몫
        if z_new <= b[2] + 1e-12:
            return a, b, True
        H = CAPST["H"].get(k, 0)
        T = np.array(a, float)
        T[0], T[2], T[3] = Qg[i, 0], z_new, Qg[i, 3]
        if k in SAW_TAB and SIG[i] >= 0:  # 밀며 썰기는 한 단계 더 감
            T[1] = SAW_TAB[k]["base_ic"][SIG[i]] + _saw_val(k, int(SIG[i]) + H + 1)
        CAPST["H"][k] = H + 1
        return a, T, False

    def cap_after(i, t, f, b, adv):
        """솔버 한 단계 뒤: 칼 자세·힘 기록, 걸러 낸 힘 갱신, 멈춤 판정."""
        CAPST["P"] = np.array(b, float)
        GATE.push_force(f[2])
        lg = CAPST["log"]
        lg["t"].append(t)
        lg["i"].append(i)
        lg["q"].append(CAPST["P"].copy())
        lg["f"].append(np.asarray(f, float).copy())
        lg["r"].append(CAPST["r"])
        lg["adv"].append(bool(adv))
        lg["ff"].append(GATE.f_filt)
        k = int(CUT_OF[i])
        c = CAPST["cuts"].get(k) if k >= 0 else None
        if c is None:
            return
        c["z_min"] = min(c["z_min"], float(b[2]))
        if str(PH[i]).startswith("cut"):
            c["fz_max"] = max(c["fz_max"], float(f[2]))
            c["fy_absmax"] = max(c["fy_absmax"], abs(float(f[1])))
            c["fz_sum"] += float(f[2])
            c["n_cut"] += 1
            c["slow"] += int(not adv)
            if not CAPST["stalled"].get(k, False):
                stall, why = GATE.note(float(b[2]), t + 1, not adv)
                if stall:
                    CAPST["stalled"][k] = True
                    CAPST["z_stall"][k] = float(b[2])
                    c.update(outcome="stall", why=why, t_stall=t + 1, z_stall=float(b[2]))
                    print(f"  힘 한계 {GATE.cap:g} N: 칼질 {k} 멈춤 — {why} (날 끝 {b[2] * 1e3:.2f} mm, 시간 {(t + 1) * step_dt:.3f} s, "
                          f"걸러 낸 힘 {GATE.f_filt:.1f} N)", flush=True)
            if adv and (i + 1 >= n_steps or PH[i + 1] != PH[i]):  # 칼질 단계 끝
                c["t_cut_end"] = t + 1
                if c["outcome"] is None:
                    c["outcome"] = "cut"


i = 0
t = 0  # 11-24 reports/65: 솔버 시간(1 ms 단계 수). 힘 한계가 없으면 늘 i(경로 줄 번호)와 같다
row_first = True  # 이 단계가 경로 줄 i 의 첫 1 ms 인가(힘 한계로 덜 내려가면 같은 줄을 여러 번 돎)
og_steps = 0
while i < n_steps:
    t_og_end = (og_steps + 1) * dt
    t0 = time.perf_counter()
    while i < n_steps and (t + 1) * step_dt <= t_og_end + 1e-9:  # 이 물리 단계에 든 1 ms 칼 기록
        a, b = Qg[i], Qg[min(i + 1, n_steps - 1)]
        adv = True  # 이 1 ms 에 경로 줄 i 를 끝내나(힘 한계로 덜 내려가면 False — 같은 줄에 머묾)
        if GATE is not None:
            a, b, adv = cap_pose(i, t, row_first)
        if row_first and args.pos_jitter > 0 and args.jitter_step > 0 and i == args.jitter_step:  # 11-23 reports/63 시험용(위 설명)
            _xj = sim.x.numpy()
            _xj += np.random.default_rng(args.jitter_seed).normal(0.0, args.pos_jitter, _xj.shape).astype(np.float32)
            sim.x.assign(_xj)
            print(f"  입자 자리 흔들기: 표준편차 {args.pos_jitter:g} m, 씨앗 {args.jitter_seed} (step {i}, {PH[i]})", flush=True)
        if row_first and PLAN2 and i == PLAN2["relocate_step"]:
            relocate(i)
        if args.knife_friction_post is not None:  # 11-17: 칼질(cut) 단계에서만 원래 칼 마찰, 그 밖에서는 post 값(다음 칼질에 되돌림)
            kf_want = kf_cut if str(PH[i]).startswith(("cut", "bsaw", "start", "settle")) else args.knife_friction_post
            if sim.material.knife_friction != kf_want:
                sim.material = dataclasses.replace(sim.material, knife_friction=kf_want)
                print(f"  칼 마찰 → {kf_want} (step {i}, {PH[i]})", flush=True)
        if args.pull_check and fixes is not None:  # 11-21 reports/60 확인 동작: pull 동안만 손이 덩이를 -x 로 끔
            fixes.hold_vel = (-PULL_V, 0.0, 0.0) if PH[i] == "pull" else (0.0, 0.0, 0.0)
        traj = KnifeTrajectory.linear(t * step_dt, (t + 1) * step_dt, solver_pose(a), solver_pose(b), begins_cut=(row_first and i in cut_steps))
        if pair_info and (sim.n_pieces != pair_np or t % 20 == 0):  # 접착 대상 = 몸통(가장 큰 조각) 뺀 조각
            _mem = np.ones(int(sim.config.max_pieces), np.int32)
            if not args.pair_adh_body:
                _mem[int(np.argmax(np.bincount(sim.pieces())))] = 0
            sim.adh_member = wp.array(_mem, dtype=int, device=args.device)
            pair_np = sim.n_pieces
        try:
            f, _ = sim.advance((t + 1) * step_dt, traj)
        except RuntimeError as e:  # 11-14: 조각 수 상한(팀원 솔버 30)에 닿으면 여기까지 저장하고 멈춤(전에는 48분 계산을 통째로 잃음)
            if "too many pieces" not in str(e):
                raise
            print(f"  조각 수 상한({sim.config.max_pieces})에 닿아 step {i}({PH[i]})에서 멈춤 — 여기까지 저장", flush=True)
            n_steps = i
            Qg, PH, forces, adh_log = Qg[:i], PH[:i], forces[:i], adh_log[:i]
            stopped_early = True
            break
        forces[i] = f  # 힘 한계 판: 같은 줄을 여러 번 돌면 마지막(줄을 끝낸) 단계 값. 단계마다 값은 log.npz cap_f
        if GATE is not None:
            cap_after(i, t, f, b, adv)
        last = adv and (i == n_steps - 1 or PH[i + 1] != PH[i])
        if fixes is not None:
            if i == settle_end:
                if args.hold_follow is not None and args.plan:
                    fixes.hold_box = hand_box(plan_cuts[0] - args.hold_follow)
                print(f"  손: 입자 {fixes.grip()}개 (step {i})", flush=True)
            elif args.hold_follow is not None and args.plan and i in follow_steps:  # 손이 다음 칼질 자리 뒤로 물러남
                fixes.hold_box = hand_box(follow_steps[i] - args.hold_follow)
                print(f"  손 물러남: 앞 끝 {(follow_steps[i] - args.hold_follow) * 1000:.1f} mm, 입자 {fixes.grip()}개 (step {i})", flush=True)
            elif PLAN2 and i == settle2_end:  # 이어 하기: 손을 쌓은 편 위로
                fixes.hold_box = hand_box2(PLAN2["cuts"][0] - args.hold_follow2)
                print(f"  손(채썰기): 입자 {fixes.grip()}개 (step {i})", flush=True)
            elif PLAN2 and i in follow2:
                fixes.hold_box = hand_box2(follow2[i] - args.hold_follow2)
                print(f"  손 물러남(채썰기): 앞 끝 {(follow2[i] - args.hold_follow2) * 1000:.1f} mm, 입자 {fixes.grip()}개 (step {i})", flush=True)
            fixes.update(t, PH[i], last, a)  # 11-24: 감쇠·넘어뜨리기 시간 재기는 솔버 시간 t(힘 한계가 없으면 t == i)
            if fixes.adhesion is not None:
                adh_log[i] = fixes.adhesion_read()
                if adh_shift_log is not None:
                    adh_shift_log[i] = fixes.adh_shift.numpy()
        if pair_log is not None:  # 조각끼리 접착: 1 ms 단계마다 읽고 비움
            _st = sim.adh_stat.numpy()
            pair_log[i] = (_st[0] * sim.dt / step_dt, _st[1] / step_dt)
            sim.adh_stat.zero_()
        if gdiag_log is not None:  # 진단: 조각끼리 맞닿음이 조각 1 에 준 충격량(자리별), 1 ms 단계마다 읽고 비움
            gdiag_log[i] = sim.gdiag.numpy()
            gcnt_log[i] = sim.gcnt.numpy()
            sim.gdiag.zero_()
            sim.gcnt.zero_()
        if args.dense_below > 0 and a[2] < args.dense_below and str(PH[i]).startswith(("cut", "dwell", "withdraw")) \
                and i % max(1, args.dense_every) == 0:  # 진단: 바닥 근처 촘촘한 기록
            dense["x"].append((sim.positions() - off).astype(np.float32))
            dense["v"].append(sim.v.numpy().astype(np.float16))
            dense["lab"].append(sim.pieces().astype(np.int8))
            dense["step"].append(i); dense["q"].append(a); dense["f"].append(f)
            ac = sim.active_cut
            dense["front"].append(sim.front.numpy()[ac].astype(np.float32) if ac >= 0 else np.full(sim.n_bins, np.nan, np.float32))
            dense["area_path"].append(sim.area_path.numpy()[ac].astype(np.float32) if ac >= 0 else np.full(sim.area_path.shape[1], np.nan, np.float32))
            dense["cut"].append(ac)
        if t % 40 == 0 or last:  # 11-24: 영상 프레임은 솔버 시간으로 고르게(힘 한계가 없으면 t == i)
            X = sim.positions() - off
            if last:
                snaps[str(PH[i])] = X.astype(np.float32)
            if t % 40 == 0:
                frames["x"].append(X.astype(np.float32))
                frames["lab"].append(sim.pieces().astype(np.int8))
                frames["step"].append(i)
                frames["t"].append(t)
                frames["q"].append(a)
                if args.saw_sever:  # 11-21: 칼질마다 남은 아래층 높이(0 = 다 잘림) [m]
                    frames["lig"].append([sim.ligament_height(c) if c < sim.n_cuts else np.nan
                                          for c in range(len(plan_cuts) + (len(PLAN2["cuts"]) if PLAN2 else 0))])  # 11-21 저녁: 이어 하기 채썰기 칼질도
        if t % 500 == 0:
            print(f"  step {i}/{n_steps} {PH[i]} z={a[2] * 1000:.1f}mm Fz={f[2]:.2f}N pieces={len(set(sim.pieces().tolist()))} "
                  f"{time.time() - t_wall:.0f}s" + (f" t={t}" if GATE is not None else ""), flush=True)
        t += 1
        row_first = adv
        if adv:
            i += 1
    mpm_s += time.perf_counter() - t0
    if args.no_og:  # 솔버만: 물리 단계 수만 센다
        og_steps += 1
        continue
    q = Qg[min(i, n_steps - 1)] if GATE is None else CAPST["P"]  # 힘 한계 판: 실제 칼 자세
    p = knife_pose_our(q)
    knife_obj.set_position_orientation(position=(p.origin + W_OFF).tolist(), orientation=knife_quat(p))
    with og.sim.render_on_step(False):
        og.sim.step()
    og_steps += 1
    if og_steps * dt >= next_frame - 1e-9:
        next_frame += frame_every
        t1 = time.perf_counter()
        points.update(to_world(sim.positions()))
        og.sim.render()
        for c, cam in cams.items():
            rgb = cam.get_obs()[0]["rgb"]
            rgb = (rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb))[:, :, :3].astype(np.uint8)
            writers[c].append_data(rgb)
            if n_frames == 0 or n_frames % (2 * args.fps) == 0:
                (out / "frames").mkdir(exist_ok=True)
                imageio.imwrite(out / "frames" / f"{c}_{n_frames:04d}.png", rgb)
        n_frames += 1
        render_s += time.perf_counter() - t1
for w in writers.values():
    w.close()

# 우리 진단이 읽는 꼴로 저장(warp_t4_replay.py 와 같음)
x_rest = (X_REST0 if PLAN2 else (sim.x0 - off)).astype(np.float32)  # 11-19: 이어 하기는 원래 처음 자리(렌더 메쉬와 맞음)
labels = np.zeros(len(x_rest), np.int64)
for c in sorted(m["x_mm"] / 1000 for m in gmeta["marks"]):
    labels += (x_rest[:, 0] > c).astype(np.int64)
meta = dict(gmeta)
if SH:
    meta.update({"shape": args.shape, "food_mesh": SH["source"], "food_xform": SH["xform"], "pre_trims": SH["pre_planes"],
                 "pre_splits": ([{"n": np.eye(3)[int(SH.get("plank_axis", 1))].tolist(), "d": float(e)} for e in SH["plank_edges"]]
                                if args.planks else []),
                 "color_in_source_frame": bool(SH.get("stack", False)),  # 11-14: 눕힌 편은 무늬(씨·껍질)를 원래 오이 좌표에서
                 "bounds": SH["bounds"]})
    if SH.get("colors"):  # 11-20 새 식재료: 겉·속 색과 무늬 함수 이름(render_surface.veg_color_fn)
        meta.update({"colors": SH["colors"], "category": SH.get("veg", args.veg), "layers": SH.get("layers"),
                     "shell": SH.get("shell"), "shell_E": args.shell_E, "shell_gc_mul": args.shell_gc_mul,
                     "shell_gc_mode": args.shell_gc_mode, "region_E": [args.region_E_mul, args.region_xmin],
                     "saw": [args.saw_amp, args.saw_hz], "no_layers": bool(args.no_layers)})
meta.update({"solver": "teammate Warp MPM" + ("" if args.no_og else " in OmniGibson (Isaac Sim) process"), "particle_size": spacing, "dx": args.dx,
             "material": dataclasses.asdict(mat), "veg": args.veg, "genesis_run": str(gr), "fixes": fixes.summary() if fixes else None})
st = np.array(frames["step"])
np.savez_compressed(out / "frames.npz", x=np.array(frames["x"]), lab=np.array(frames["lab"]), step=st,
                    t=(st * step_dt if GATE is None else np.array(frames["t"]) * step_dt),  # 11-24: 힘 한계 판은 솔버 시간
                    q=np.array(frames["q"]), phase=PH[st], x_rest=x_rest, mat=np.zeros(len(x_rest), np.int8),
                    **({"lig": np.array(frames["lig"], float)} if frames["lig"] else {}),
                    meta=json.dumps(meta, ensure_ascii=False, default=float))
np.savez_compressed(out / "log.npz", q=Qg, phase=PH, labels=labels, f=forces, adh_og=adh_log, **{f"pos_{k}": v for k, v in snaps.items()},
                    **({"adh_shift": adh_shift_log[:len(PH)]} if adh_shift_log is not None else {}),
                    **({"gdiag": gdiag_log[:len(PH)], "gcnt": gcnt_log[:len(PH)]} if gdiag_log is not None else {}),
                    **({"pair_adh": pair_log[:len(PH)]} if pair_log is not None else {}),
                    **({f"cap_{_k}": np.array(_v) for _k, _v in CAPST["log"].items()} if GATE is not None else {}))
if dense["step"]:  # 진단(--dense_below): 솔버 좌표 off 와 칼 틀(cut_o·cut_u·cut_v·cut_w), 칸 시작 bin_umin 도 함께
    np.savez_compressed(out / "dense.npz", **{k: np.array(v) for k, v in dense.items()}, off=np.asarray(off, float),
                        cut_o=sim.cut_o.numpy(), cut_u=sim.cut_u.numpy(), cut_v=sim.cut_v.numpy(), cut_w=sim.cut_w.numpy(),
                        bin_umin=float(sim.bin_umin), dx=float(sim.dx), board_z=float(sim.board_z), dilation=float(sim._knife.dilation))
res = {"veg": args.veg, "wall_s": round(time.time() - t_wall, 1), "solver_wall_s": round(mpm_s, 1), "render_wall_s": round(render_s, 1),
       "frames": n_frames, "n_particles": int(sim.n_particles), "dt_s": float(sim.dt), "physics_hz": hz,
       "pieces_final": int(len(set(sim.pieces().tolist()))), "cut_complete": [bool(sim.cut_complete(c)) for c in range(sim.n_cuts)],
       "warnings": sim.warnings, "no_og": args.no_og, "knife_floor_m": args.knife_floor, "blade_split": bool(args.blade_split), "knife_side": bool(args.knife_side), "gap_contact": args.gap_contact, "face_tan": bool(args.face_tan), "reside": bool(args.reside), "pinch_extra": args.pinch_extra, "dilation_cells": args.dilation, "end_damp": [args.end_damp, args.end_h, args.end_scope], "knife_bevel": args.knife_bevel, "release_outer": bool(args.release_outer),
       "knife_friction_post": args.knife_friction_post, "face_contact": args.face_contact, "adh_anchor": bool(args.adh_anchor),
       "kerf_slip": bool(args.kerf_slip), "kerf_depth_cells": args.kerf_depth, "pair_adh": pair_info,
       "knife": gmeta["knife"], "plan_mm": gmeta.get("plan_mm"), "stopped_early": stopped_early, "wet": args.wet, "length": V["L"], "top_margin": args.top_margin, "max_resist_N": [float(forces[PH == f"cut{c}", 2].max()) for c in range(sim.n_cuts)],
       "material": dataclasses.asdict(mat), "fixes": fixes.summary() if fixes else None, "warp": wp.__version__,
       "plan2": (dict(PLAN2, relocate=RELOC) if PLAN2 else None), "cfl": float(sim.config.cfl),
       "fast_box": int(getattr(sim, "fast_box", 0)), "fast_diag": int(getattr(sim, "fast_diag", 0)), "fast_launch": int(getattr(sim, "fast_launch", 0)),
       "fast_py": int(getattr(sim, "fast_py", 0)), "fast_fuse": int(getattr(sim, "fast_fuse", 0)), "fast_gmask": int(getattr(sim, "fast_gmask", 0)), "pos_jitter": float(args.pos_jitter), "jitter_seed": int(args.jitter_seed), "jitter_step": int(args.jitter_step), "poison": int(getattr(sim, "fast_poison", 0)), "wp_det": int(args.wp_det), "wp_nofma": int(args.wp_nofma), "nan_hits": int(getattr(sim, "nan_hits", 0)), "pc_hits": int(getattr(sim, "pc_hits", 0)), "fast_sel2": bool(args.fast_sel2),
       "pc_frac_mean": (float(np.mean(sim.pc_cells)) if getattr(sim, "pc_cells", None) else None),
       "box_violations": int(getattr(sim, "box_violations", 0)),
       "box_frac_mean": (float(np.mean(sim.box_cells)) if getattr(sim, "box_cells", None) else None),
       "fast3_auto": bool(FAST3_AUTO), "fast3_solver": FAST3_PATH, "fast3_note": FAST3_NOTE,
       "solver_dir": str(Path(sys.modules[MPMCuttingSim.__module__].__file__).parents[2])}
res["force_cap"] = None
if GATE is not None:  # 11-24 reports/65: 칼질마다 썰림/멈춤·늦어짐·닿은 깊이·힘
    _cuts = []
    for _k in sorted(CAPST["cuts"]):
        _c = CAPST["cuts"][_k]
        _zt = float(ZT) if _k < len(plan_cuts) else float(PLAN2["z_top_plan"]) if PLAN2 else float(ZT)
        _cuts.append({"k": _k, "outcome": _c["outcome"] or "unfinished", "why": _c.get("why"),
                      "t0_s": _c["t0"] * step_dt, "t_cut_end_s": (_c["t_cut_end"] * step_dt if "t_cut_end" in _c else None),
                      "t_stall_s": (_c["t_stall"] * step_dt if "t_stall" in _c else None), "slow_s": _c["slow"] * step_dt,
                      "z0_mm": _c["z0"] * 1e3, "z_min_mm": _c["z_min"] * 1e3, "z_stall_mm": (_c["z_stall"] * 1e3 if "z_stall" in _c else None),
                      "food_top_mm": _zt * 1e3, "depth_frac": float(np.clip((_zt - _c["z_min"]) / max(_zt - 0.0005, 1e-9), 0.0, 1.0)),
                      "fz_max_N": _c["fz_max"], "fz_mean_N": (_c["fz_sum"] / _c["n_cut"] if _c["n_cut"] else None),
                      "fy_absmax_N": _c["fy_absmax"], "cut_complete": (bool(sim.cut_complete(_k)) if _k < sim.n_cuts else None)})
    res["force_cap"] = dict(GATE.summary(), arg=args.force_cap, sim_steps=int(t), plan_steps=int(n_steps),
                            saw=[args.saw_amp, args.saw_hz], cuts=_cuts)
    for _c in _cuts:
        print(f"  힘 한계 {GATE.cap:g} N 칼질 {_c['k']}: {_c['outcome']}, 깊이 {_c['depth_frac']:.2f}, 늦어짐 {_c['slow_s']:.2f} s, "
              f"최대 {_c['fz_max_N']:.1f} N, 잘림 {_c['cut_complete']}", flush=True)
if FAST3_NOTE and FAST3_AUTO:  # 11-24: 기본 빠른3 을 켰는데 못 썼으면 경고 목록에도(OmniGibson 판처럼 처음부터 안 켠 것은 fast3_note 에만)
    res["warnings"] = list(res.get("warnings") or []) + [f"fast3 default not used: {FAST3_NOTE}"]
if res["box_violations"]:  # 11-24: 빠른 판 상자 벗어남(1 ms 동안 입자가 둘레 여유보다 멀리 감)은 결과가 틀릴 수 있다는 뜻
    res["warnings"] = list(res.get("warnings") or []) + [f"fast box violations: {res['box_violations']} (결과 의심, --no_fast3 로 다시 견줄 것)"]
try:  # 11-21 reports/58: 칼질마다 균열 장부(칼의 일로 연 균열·도마 끼임으로 공짜로 연 균열·찢김) — 질김이 어디서 막히는지 보려고
    import simfoundry.cutting.mpm as _mpm
    res["crack_ledger"] = {"names": list(_mpm.LEDGER_NAMES), "rows": sim.crack_ledger.numpy().tolist()}
    res["area_path"] = {"cols": ["energy", "pinch", "tear", "saw"][:sim.area_path.shape[1]], "rows": sim.area_path.numpy().tolist()}
    res["connected_pieces"] = int(sim.connected_pieces())  # 11-21 reports/59: 아래층 등으로 아직 붙어 있는 조각은 한 덩어리로 셈
    if args.saw_sever:  # 11-21 reports/59: 칼질마다 칸별 미끄러진 거리 최대·중앙(재료가 있던 칸)
        _tr = sim.saw_travel.numpy()[:sim.n_cuts]
        _nd = sim.saw_need.numpy()[:sim.n_cuts] if hasattr(sim, "saw_need") else None
        res["saw_sever"] = {"h_m": float(sim.saw_h), "len_m": float(sim.saw_len), "bottom_saw_s": args.bottom_saw_s,
                            "region": args.saw_region,
                            # 11-21 reports/60: 칼질마다 질긴 층을 둔 칸(날 방향 위치 u [m], 칸 가운데) — 자리가 일부일 때 어디였나
                            "tough_bins_u_m": ([[float(sim.bin_umin + (b + 0.5) * sim.dx) for b in np.flatnonzero(r > 0)] for r in _nd]
                                               if _nd is not None else None),
                            "travel_max_m": [float(r.max()) for r in _tr],
                            "travel_median_m": [float(np.median(r[r > 0])) if (r > 0).any() else 0.0 for r in _tr]}
except Exception as _e:  # noqa: BLE001
    res["crack_ledger"] = f"못 읽음: {_e}"
(out / "metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
print(json.dumps({k: res[k] for k in ("wall_s", "solver_wall_s", "render_wall_s", "frames", "n_particles", "pieces_final",
                                      "cut_complete", "max_resist_N", "warp")}, ensure_ascii=False), flush=True)
os._exit(0)  # OmniGibson 종료 정리가 느리거나 멈출 때가 있다(팀원 스크립트와 같은 이유)
