"""팀원 파이프라인(OmniGibson + Franka)에서 우리 썰기 물리로 학습 에피소드를 만든다 (2026-11-24, reports/66, 계획 docs/plan_2026-11-24_force_pipeline_grasp.md 4절).

  OMNIGIBSON_HEADLESS=1 PYTHONPATH=$HOME/innerinner_datagen \\
  python scripts/cutting/food/food_robot_episodes.py --out <폴더> --episodes 1 -- \\
      --veg cucumber <og_veg_cut.py 의 식재료·물리·경로 인자 … --plan 3,3,3,3 … [--force_cap elrobot_motor]>

무엇
- 물리: 우리 og_veg_cut.py 의 준비 부분(식재료 모양·재료·솔버 고침·손·칼 접착·칼 경로·힘 한계 인자)을 이 프로세스에서 그대로 돌려 쓴다
  ('실행기 본체' 표시 앞까지. --no_og 를 붙임. 11-25 부터 기본 = 빠른3: Warp 1.8 호환 빠른3 복사본 <빠른3 솔버 복사본>
  (patches/build_datagen_fast3.sh)을 먼저 불러오고 --fast3 을 붙임 — reports/71. `--no_fast3` 이면 원래 솔버·--no_fast3.
  시범 2판(reports/70)을 그대로 다시 하려면 `--no_fast3 --knife_grasp current --move_pick slice`(+ 첫 틱 4번 그리기는 코드 차이).
  솔버는 PYTHONPATH 의 팀원 작업 브랜치 이 저장소(research4/food-cutting)에서 온다.
- 로봇·장면·기록: 팀원 scripts/cutting/tofu_og_cut.py(T: Arm, grasp_knife, move_knife, arm_controller_config)와
  tofu_policy_episodes.py(P: EpisodeRun.capture 의 front·wrist 224×224·state 16·cmd 9 기록, capped_coupler 손에 거는 힘 50 N·5 N·m 상한),
  simfoundry.cutting.cosim.KnifeCoupling + og_cosim.PhysicsStepCoupler(물리 단계 뒤 로봇이 쥔 칼 자세로 솔버 진행, 다음 단계 전 칼에 힘).
  장면은 팀원 재구성 장면(Data/tofu_chefd, 이 컴퓨터에 없음) 대신 우리 빈 장면: 탁자 0.76 m·도마 18 mm(충돌체)·칼(강체)·Franka(assisted 잡기, IK).
- 칼 경로: 우리 --plan 경로(1 ms 줄)를 물리 단계(120 Hz)마다 단계 길이만큼 따라간다(PlanFollower). 칼질 중에는 --force_cap 몫만큼만
  (force_cap.ForceGate, 칼이 받은 위쪽 힘 = 결합기 기록). 밀며 썰기(--saw_amp)는 시간에 따라. 손 상자·칼 마찰·순간 이동 쌓기(--plan2) 같은
  경로 줄 사건은 og_veg_cut.py 와 같은 줄에서.
- 내보내기: 에피소드마다 ep_XXXXX.npz·.json(팀원 꼴 그대로 → 팀원 scripts/cutting/policy/npz_to_lerobot.py, 과제 문장은 감싸개
  scripts/cutting/food/food_npz_to_lerobot.py 로) + ep_XXXXX_phys/frames.npz·log.npz·metrics.json(우리 진단·메쉬 렌더용, og_veg_cut.py 와 같은 꼴).

한계(11-24)
- 화면의 식재료(11-25, 계획 B5, reports/69): 기본 `--food_view mesh` = 껍질·단면 메쉬(에피소드 전에 food_mesh_prep.py 가 계획된 칼질 평면으로
  미리 나눔 → food_mesh_view.FoodMesh 가 화면 찍기 직전마다 입자를 따라 옮김, 강체로 넘긴 뒤에는 강체를 따라감). `--food_view points` = 팀원 입자 공(FoodPoints).
- 솔버는 빠른3 이 아니라 원래 속도(같은 프로세스 Isaac Sim Warp 1.8). 칼 잡기는 팀원 방식(닿으면 붙여 줌·칼 중력 끔).
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # og_veg_cut.py 와 같은 까닭(reports/62: numpy 일꾼 스레드가 주 스레드를 방해)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OGVC = HERE / "og_veg_cut.py"
MARK = "# ==== 실행기 본체: 여기부터 og_veg_cut.py 의 칼 경로 실행"


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--physics_hz", type=float, default=120.0)
    ap.add_argument("--policy_hz", type=float, default=10.0)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--wrist", type=float, nargs=3, default=[0.06, 0.0, -0.10], help="손목 카메라(손끝 틀) [m] — 팀원 기본")
    ap.add_argument("--standoff", type=float, default=0.52, help="로봇 받침을 칼 손잡이 평균 자리에서 +y 로 이만큼 [m] — 팀원 기본")
    ap.add_argument("--wrench_cap", type=float, nargs=2, default=[50.0, 5.0], help="손(칼)에 거는 재료 힘·돌림힘 상한 — 팀원 기본")
    ap.add_argument("--robot_below_table", type=float, default=0.06, help="로봇 받침 높이 = 탁자 높이 − 이 값 [m](받침대 위)")
    ap.add_argument("--shift", type=float, default=0.0, help="에피소드마다 식재료와 칼 경로를 도마 위에서 x·y 로 ±이만큼 옮김 [m]")
    ap.add_argument("--yaw_deg", type=float, default=0.0, help="에피소드마다 식재료와 칼 경로를 도마 위에서 ±이만큼 돌림 [도]")
    ap.add_argument("--wet_lognormal", type=float, nargs=2, default=None, metavar=("MEDIAN", "SIGMA"),
                    help="에피소드마다 --wet 를 로그정규로 뽑음(reports/52 추천 0.837 0.228). 안 주면 식재료 인자 그대로")
    ap.add_argument("--overview", type=int, nargs=2, default=None, metavar=("W", "H"), help="영상용 넓은 카메라(학습 입력 아님)")
    ap.add_argument("--overview_fps", type=float, default=10.0)
    ap.add_argument("--task", default=None, help="과제 문장(기록만; 기본 'slice the <식재료> into <두께> mm slices')")
    ap.add_argument("--approach_ang_speed", type=float, default=0.4)
    ap.add_argument("--max_ep_s", type=float, default=180.0, help="에피소드 한 판 칼 경로의 최대 시간(안전장치) [s]")
    ap.add_argument("--task_kind", default="cut", choices=["cut", "place", "move", "move_replay", "unstick_replay"],
                    help="cut = 칼을 쥐고 썰기(기본), place = 빈 손으로 통 식재료(강체)를 탁자에서 집어 도마에 놓기(과제 ②), "
                         "move = 썰기 뒤 조각을 강체로 바꾸고(기록 밖에서 칼 치움) 조각 하나를 접시로 옮기기(과제 ③, <out>/move/ 에 따로 기록)")
    ap.add_argument("--to_rigid", action="store_true",
                    help="썰기 에피소드가 끝나면 모든 조각을 PhysX 강체로 바꾸고(food_handoff.pieces_to_rigid) 1 초 둬서 가만히 있는지 잼(단계 C1 확인)")
    ap.add_argument("--food_view", default="mesh", choices=["mesh", "points"],
                    help="화면의 식재료: mesh = 껍질·단면 메쉬(11-25 B5, 기본), points = 팀원 입자 공(FoodPoints, 11-24 판)")
    ap.add_argument("--mesh_tex", type=int, default=512, help="단면 질감 한 변 화소(food_mesh_prep.py --tex)")
    ap.add_argument("--mesh_residual", type=float, default=0.6, help="강체 맞춤에서 남는 변형을 겉면에 옮기는 비율(og_mesh_render 와 같은 기본)")
    ap.add_argument("--track_mm", type=float, nargs=2, default=[0.5, 1.5], metavar=("LO", "HI"),
                    help="11-25: 실제 칼이 명령보다 LO mm 넘게 위면 칼질·바닥 머무름 경로를 늦추고 HI mm 면 --track_floor 빠르기까지 늦춤(팔 기다리기). 0 0 = 끔. "
                         "0.5·1.5 = m5 에서 시험한 값(칼 바닥 4.1 → 1.38 mm). 0.3·1.0 은 칼이 힘을 받아 늘 1 mm 넘게 떠 있는 자세에서 경로가 서 버려 '멈춤'이 됨(2판 첫 시도)")
    ap.add_argument("--track_floor", type=float, default=0.15,
                    help="팔 기다리기로 늦출 때 가장 느린 빠르기 몫(0 이면 아예 섬 — 칼이 힘을 받아 못 내려가는 자세에서 경로가 영영 멈춤)")
    ap.add_argument("--prep_python", default=sys.executable,
                    help="메쉬 미리 만들기(food_mesh_prep.py)를 돌릴 파이썬(manifold3d·open3d 가 있는 cutsim 환경)")
    ap.add_argument("--move_pick", default="auto", choices=["auto", "body", "slice"],
                    help="11-25 reports/72: 옮기기 과제에서 집을 조각 고르기(plan_piece_grasp). auto = 손가락 길이 막히지 않은 선 조각 → 몸통, "
                         "slice = 11-24 규칙(선 편, 이웃·몸통에 손가락이 걸려 당근·양파 0/3)")
    ap.add_argument("--tip_pinch", action="store_true", help="(기본 켬 — 남겨 둔 옛 인자)")
    ap.add_argument("--no_tip_pinch", action="store_true",
                    help="11-25 reports/77: 누운 얇은 편 집기 끄기. 기본은 켬 — 손끝 가까이 잡는 광선 점을 더하고(TIP_SHIFT 7.5 mm) 손끝을 도마 위 0.6 mm 로")
    ap.add_argument("--tip_ray_h", type=float, default=0.0021, help="누운 편을 집을 때 손끝 잡는 광선 높이(도마 위) [m]")
    ap.add_argument("--glue", default="on", choices=["on", "off"],
                    help="11-25 reports/77: 썰기 끝에 칼에 붙어 있던 조각을 강체 단계에서도 칼에 붙여 둠(끊어지는 이음목, 버팀 = film3 정지 미끄럼 버팀 × 넓이)")
    ap.add_argument("--unstick", default="tap", choices=["shake", "tap", "scrape", "wipe", "off"],
                    help="칼에 붙은 편 떼기 과제(<out>/unstick/ 에 따로 기록). 기본 tap(11-26 사용자 결정) = 칼끝을 숙여 도마에 내리치기, "
                         "shake = 접시 위에서 위아래로 털기(11-25 reports/77, 11-26 전까지 기본), scrape = 칼을 기울여 깊은 네모 그릇 벽 모서리에 대고 끌어올려 긁기, "
                         "wipe = 두 번째 팔(도움 팔)이 집게를 벌려 칼날을 끼우고 쓸어내리기(11-25 밤 reports/78)")
    ap.add_argument("--glue_model", default="slide", choices=["slide", "weld"],
                    help="11-25 밤 reports/78: 칼에 붙여 둔 편의 접착 모형. slide = D6 이음(칼 면을 따라 버팀까지 용수철, 넘으면 미끄러짐·넓이 줄면 버팀 줄음 — "
                         "관성·손가락·접시 벽·도마 충격을 PhysX 가 같은 식으로 풂), weld = 77 판(고정 이음 + 칼 가속도 관성 하중 문턱)")
    ap.add_argument("--slide_fn", type=float, default=25.0, help="slide 이음 고유 진동수 [Hz](쥔 칼의 미세 떨림 거르기)")
    ap.add_argument("--tap", type=float, nargs=4, default=[0.5, 8, 12.0, 0.05], metavar=("SPEED", "TAPS", "PITCH_DEG", "START_H"),
                    help="tap: 칼끝을 PITCH_DEG 숙이고(칼 뒤쪽·편 자리 날 끝은 도마 위에 뜸) START_H [m] 위에서 SPEED [m/s] 로 내려 칼끝으로 도마를 침 × TAPS "
                         "(다 떨어지면 멈춤). 붙은 편 쪽 도마가 빈 곳으로 칼을 옮겨서 침")
    ap.add_argument("--scrape", type=float, nargs=3, default=[30.0, 0.04, 0.0005], metavar=("TILT_DEG", "SPEED", "GAP"),
                    help="scrape: 편이 붙은 면이 아래를 보게 칼을 TILT_DEG 기울여 네모 접시 벽 안쪽 위 모서리에 GAP [m] 띄워 대고, 칼등 쪽으로 SPEED [m/s] 끌어올림 "
                         "→ 모서리가 칼 면을 칼등에서 날 끝 쪽으로 훑어 편을 접시 안으로 밀어냄")
    ap.add_argument("--tray_gap", type=float, default=0.07,
                    help="scrape 네모 접시: 긴 벽의 +y(로봇·손잡이) 끝 이만큼 [m]은 낮은 턱 — 칼 뒤꿈치를 높은 벽 끝에 맞추면 손잡이·손은 턱 위, "
                         "뒤꿈치 쪽 편은 접시 안에 떨어짐(칼날이 8 cm 뿐이고 편이 뒤꿈치 쪽에 붙음, 11-25 밤 s4~s7)")
    ap.add_argument("--tray", type=float, nargs=4, default=[0.18, 0.26, 0.10, 0.008], metavar=("X", "Y", "WALL_H", "END_H"),
                    help="scrape 의 네모 접시 [m]: 크기 X·Y, 칼과 나란한 쪽(±x) 벽 높이 WALL_H, 다른 쪽(±y) 턱 높이 END_H(바닥·벽 두께 6 mm). "
                         "긁는 벽(Y)은 칼날보다 짧게 — 칼 뒤꿈치·손잡이(칼날보다 두꺼움)가 벽 끝 밖에 있어야 칼 면을 벽 모서리에 댈 수 있음(11-25 밤 s1~s3: "
                         "손잡이·칼이 벽에 걸려 26~36 mm 앞에서 멈춤). 짧은 쪽은 낮은 턱이라 기울여 내려간 날이 안 걸림")
    ap.add_argument("--wipe", type=float, nargs=3, default=[0.05, 0.001, 0.018], metavar=("SPEED", "GAP", "STEP"),
                    help="wipe: 도움 팔 손끝(닫은 손가락)을 칼 면에서 GAP [m] 띄워 가장 높은 편 위에서 날 끝 가까이까지 SPEED [m/s] 로 쓸어내림, "
                         "붙은 편 폭이 넓으면 칼 길이 방향 STEP [m] 마다 한 번씩")
    ap.add_argument("--bowl", type=float, nargs=2, default=None, metavar=("RADIUS", "WALL_H"),
                    help="11-26 reports/79: 둥근 접시 대신 둥근 그릇(바깥 반지름·벽 높이 [m], 예 0.085 0.035) — 옮기기 과제의 놓을 곳")
    ap.add_argument("--move_n", type=int, default=1, help="11-26 reports/79: 썰기 판에서 옮길 조각 수(하나씩 차례로 그릇·접시로, 옮길 것이 없으면 멈춤)")
    ap.add_argument("--video_all", action="store_true",
                    help="11-26 reports/79: 영상용 넓은 카메라(--overview)를 썰기 → 떼기 → 옮기기 내내 찍음(썰기는 식재료 가까이, 옮기기는 도마·그릇이 함께 보이게), "
                         "편 쌓기(순간 이동) 때는 옮기는 장면을 끼워 넣음, 국면 시간은 <out>/video_marks.json")
    ap.add_argument("--plate_xy", type=float, nargs=2, default=None, help="접시 자리 [m](기본 −0.25 0.22, wipe 는 도움 팔 쪽 0.27 0.22)")
    ap.add_argument("--helper_xy", type=float, nargs=2, default=[0.27, -0.56],
                    help="wipe 도움 팔(두 번째 Franka) 받침 자리 [m] — 접시 쪽을 봄. 기본 = 탁자 앞(칼끝 쪽): 옆(+x, 0.70 0.22)에 두면 두 팔 손목이 "
                         "손잡이 위에서 붙어 칼이 23~74 mm 밀림(11-25 밤 w3~w9)")
    ap.add_argument("--shake", type=float, nargs=4, default=[0.03, 5.0, 3, 3], metavar=("AMP", "HZ", "CYCLES", "ROUNDS"),
                    help="털기: 아래로 AMP [m] 내렸다 올리기를 HZ 로 CYCLES 번 = 한 판, 다 떨어질 때까지 ROUNDS 판까지(판마다 진폭 +20 %%). "
                         "명령 최대 가속 = AMP/2·(2π·HZ)² — 기본 30 mm·5 Hz = 14.8 m/s²(Franka 사양 13 m/s² 보다 조금 큼, 11-25 reports/77)")
    ap.add_argument("--separate", default="push", choices=["push", "off"],
                    help="11-25 reports/77: 몸통에 붙어 줄지어 선 편을 칼로 밀어 떼어 놓기(옮기기 전, <out>/separate/ 에 따로 기록)")
    ap.add_argument("--replay", nargs="+", default=None,
                    help="--task_kind move_replay: 썰기 에피소드 물리 폴더(ep_*_phys, rigid_pieces.json·npz·pieces/*.usd)들 — 썰기 없이 "
                         "잘린 뒤 강체 조각을 그 자리에 다시 놓고 옮기기만(같은 식재료 인자·seed0 으로 로봇 자리가 같아짐)")
    ap.add_argument("--knife_grasp", default="canonical", choices=["canonical", "current"],
                    help="11-25 reports/73: 칼 쥐기 전 손 자세. canonical = 늘 같은 자세(손끝 아래·손 x 가 받침에서 멀어지는 쪽)로 옮긴 뒤 쥠"
                         "(칼–손 관계가 에피소드마다 같음), current = 2판까지(그때 손 방향 그대로 — 첫 판 25° 기욺·뒤 판 뒤집힘이 섞임)")
    ap.add_argument("--fast3", action="store_true",
                    help="11-25 reports/71: 솔버를 빠른3 복사본(--fast3_path, patches/build_datagen_fast3.sh 가 만듦 — Warp 1.8 호환)으로. "
                         "복사본 도장이 지금 작업 트리 simfoundry 와 다르면 경고 뒤 원래 솔버. 11-25 07:25 부터 기본(사용자 11-24 '빠른3 기본' 결정과 맞춤)")
    ap.add_argument("--no_fast3", action="store_true", help="원래 솔버(팀원 작업 트리 그대로, 2판까지)")
    ap.add_argument("--fast3_path", default=os.environ.get("SIMFOUNDRY_FAST3_PATH", ""), help="빠른3 솔버 복사본(없으면 이 브랜치 솔버 그대로)")
    ap.add_argument("--thick_choices", default=None,
                    help="11-26 reports/81: 에피소드마다 지정 두께를 이 목록(mm, 쉼표)에서 seed 로 골라 --plan 을 'n 번 같은 두께'로 바꿈(과제 문장·판정도 그 두께). 예: 3,5,8")
    ap.add_argument("--thick_n", type=int, default=4, help="--thick_choices 의 칼질 수 상한(식재료 길이 − 16 mm 안에 들게 줄임 — 손이 잡을 몸통을 남김)")
    ap.add_argument("--look", default="base", choices=["base", "real", "random"],
                    help="11-26 reports/81: 화면 겉모습(food_look.py, 물리는 그대로). base = 지금까지와 같음, real = 팀원 주방에 맞춘 한 벌"
                         "(흰 플라스틱 도마·화강암 조리대·타일 벽·천장 조명 그림자·실내 사진 돔), random = 에피소드마다 무늬·조명·돔·앞 카메라 자리를 뽑음")
    ap.add_argument("--look_seed", type=int, default=None, help="--look random 의 seed(없으면 에피소드 seed)")
    ap.add_argument("--look_preview", type=int, default=0,
                    help="11-26 reports/81: > 0 이면 첫 에피소드 첫 틱에서 base·real·random(seed 0..N−1) 겉모습으로 front·wrist(·overview) 를 찍어 "
                         "<out>/look_preview/ 에 저장하고 끝냄(시험용 — 에피소드 안 만듦)")
    ap.add_argument("food", nargs=argparse.REMAINDER, help="-- 뒤: og_veg_cut.py 인자(--veg … --plan … [--force_cap …])")
    a = ap.parse_args()
    a.fast3 = (not a.no_fast3) and bool(a.fast3_path)  # 11-25: 빠른3 기본(복사본 경로 --fast3_path·SIMFOUNDRY_FAST3_PATH 가 있을 때만)
    a.tip_pinch = not a.no_tip_pinch  # 11-25 reports/77: 누운 편 손끝 집기 기본
    food = a.food[1:] if a.food and a.food[0] == "--" else a.food
    if "--veg" not in food or "--plan" not in food:
        ap.error("-- 뒤에 og_veg_cut.py 인자가 필요(--veg, --plan 은 꼭)")
    return a, food


def food_meta(g, spacing):
    """식재료 메쉬를 만드는 데 드는 meta(og_veg_cut.py frames.npz 의 meta 중 모양·칼질·색 칸)."""
    meta = {k: v for k, v in g["gmeta"].items() if k in ("marks", "knife_roll_deg", "z_lift", "plan_mm", "colors", "category")}
    P2 = g.get("PLAN2")
    if P2:  # 11-26 reports/79: 이어 하기 채썰기 평면(쌓은 자리의 x = cx)을 처음 자리(눕히기 전)로 되돌리면 모든 편에서 같은 높이의 수평면
        #   (눕힘 R_LAY: 새 x = −처음 z + 상수, 편의 +x 끝 = stack_x 에 맞춤 → 처음 z = 편 바닥 + 두께 누적). 미리 나누기는 온 메쉬에 걸리지만
        #   안 갈린 조각은 같은 입자를 따라 붙은 채 그려져 보이지 않음(몸통·안 쌓은 편).
        zmin = float(np.min(np.asarray(g["X_REST0"], float)[:, 2]))
        cum = np.cumsum(P2["th"]) / 1000.0
        mk2 = []
        for mk in meta.get("marks", []):
            ph_ = str(mk.get("phase", ""))
            k_ = int(ph_[3:]) if ph_.startswith("cut") and ph_[3:].isdigit() else -1
            if k_ >= P2["k0"]:
                zp = zmin - 0.5 * spacing + float(cum[k_ - P2["k0"]])
                mk = {"phase": ph_, "x_mm": 0.0, "y_mm": 0.0, "z_mm": 1e3 * zp, "nrm": [0.0, 0.0, 1.0]}
            mk2.append(mk)
        meta["marks"] = mk2
    meta.update({"particle_size": float(spacing), "veg": g["args"].veg, "solver": "teammate Warp MPM"})
    SH = g.get("SH")
    if SH:
        meta.update({"shape": g["args"].shape, "food_mesh": SH["source"], "food_xform": SH["xform"], "pre_trims": SH["pre_planes"],
                     "pre_splits": [], "color_in_source_frame": bool(SH.get("stack", False)), "bounds": SH["bounds"]})
        if SH.get("colors"):
            meta.update({"colors": SH["colors"], "category": SH.get("veg", g["args"].veg), "layers": SH.get("layers"), "shell": SH.get("shell")})
    return meta


def prep_mesh(x_rest, meta, cache_root, tex, no_cuts, python):
    """food_mesh_prep.py 를 다른 파이썬(cutsim)으로 돌려 조각 메쉬를 만든다. 같은 입자·칼질·질감이면 캐시를 다시 씀. → 메쉬 폴더."""
    import hashlib
    import subprocess

    x_rest = np.ascontiguousarray(np.asarray(x_rest, np.float32))
    key_src = json.dumps({k: meta.get(k) for k in ("marks", "food_mesh", "food_xform", "pre_trims", "category", "colors", "particle_size")},
                         sort_keys=True, default=float) + f"|{tex}|{no_cuts}"
    rs_py = next(p / "scripts" / "render_surface.py" for p in HERE.parents if (p / "scripts" / "render_surface.py").exists())
    code = (HERE / "food_mesh_prep.py").read_bytes() + rs_py.read_bytes()  # 만드는 코드가 바뀌면 캐시를 새로
    key = hashlib.sha1(x_rest.tobytes() + key_src.encode() + code).hexdigest()[:12]
    mdir = Path(cache_root) / key
    if (mdir / "mesh.json").exists():
        return mdir, 0.0
    mdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mdir / "src.npz", x_rest=x_rest, meta=json.dumps(meta, ensure_ascii=False, default=float))
    env = {k: os.environ[k] for k in ("HOME", "USER", "LANG", "LC_ALL") if k in os.environ}
    env.update(PATH="/usr/bin:/bin", OPENBLAS_NUM_THREADS="8", OMP_NUM_THREADS="8")  # Isaac Sim 의 PYTHONPATH·LD_LIBRARY_PATH 를 넘기지 않음
    t0 = time.time()
    cmd = [python, str(HERE / "food_mesh_prep.py"), str(mdir / "src.npz"), "--out", str(mdir), "--tex", str(tex)] + (["--no_cuts"] if no_cuts else [])
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (mdir / "prep.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0 or not (mdir / "mesh.json").exists():
        raise RuntimeError(f"메쉬 미리 만들기 실패({r.returncode}): {(r.stdout + r.stderr)[-800:]}")
    return mdir, time.time() - t0


def col_thickness(P, spacing, cell=1e-3):
    """11-26 reports/81: 편 두께 = (y, z) 1 mm 칸마다 처음 자리 x 폭 + 입자 간격의 가운데값 [m]. 칸에 입자 3개 미만은 뺌."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return None
    key = np.floor(P[:, 1] / cell).astype(np.int64) * 100000 + np.floor(P[:, 2] / cell).astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks, xs = key[order], P[order, 0]
    cuts = np.flatnonzero(np.diff(ks)) + 1
    th = [g.max() - g.min() + spacing for g in np.split(xs, cuts) if len(g) >= 3]
    return float(np.median(th)) if th else None


def slice_report(x_rest, labels, cuts_x, spacing, plan_mm, tol_mm=2.0):
    """11-25: 계획한 편마다 '몸통과 떨어졌나'와 두께. 편 k = 처음 자리 x 가 칼질 평면 k 와 k−1(첫 편은 식재료 끝) 사이인 입자
    (평면에서 입자 간격 절반 안 층은 뺌 — 어느 쪽으로 갈지 재료마다 다름, reports/69). 솔버 라벨(칼질이 만든 조각 번호)의 가장 흔한 값이
    몸통의 것과 다르면 떨어진 것. 두께 = 그 라벨 입자들의 처음 자리 x 폭 + 입자 간격(실제로 어디가 잘렸는지; 나중에 휘거나 움직여도 같음).
    두 편이 한 라벨이면(둘 사이가 안 잘림) 두께가 두 배로 나와 ±tol 밖이 된다."""
    x = np.asarray(x_rest, float)[:, 0]
    lab = np.asarray(labels)
    band = 0.5 * spacing
    cx = sorted((float(c) for c in cuts_x), reverse=True)  # 끝(+x)에서 안쪽으로
    if not cx:
        return []
    body = x < cx[-1] - band
    lb = int(np.bincount(lab[body]).argmax()) if body.any() else -1
    out = []
    hi = x.max() + spacing
    for k, c in enumerate(cx):
        sel = (x > c + band) & (x < hi - band)
        hi = c
        if not sel.any():
            out.append({"k": k, "separated": False, "thick_ok": False, "n": 0})
            continue
        cnt = np.bincount(lab[sel])
        lk = int(cnt.argmax())
        sep = lk != lb
        th = th_col = None
        if sep:
            m = lab == lk
            th = float(x[m].max() - x[m].min() + spacing)
            th_col = col_thickness(np.asarray(x_rest, float)[m], spacing)
        want = float(plan_mm[k]) if plan_mm and k < len(plan_mm) else None
        # 11-26 reports/81: 판정은 칸별 두께(튀어나온 이웃 층 2 % 남짓에 안 흔들림). 예전 x 폭은 thickness_extent_mm 로 남김
        #   (3판 당근 편 2~4 가 x 폭 4.3~5.2 mm 로 실패였으나 칸별 두께는 2.5~2.6 mm — 오이·양파 편과 같음)
        ok = bool(sep and want is not None and th_col is not None and abs(1e3 * th_col - want) <= tol_mm)
        out.append({"k": k, "separated": bool(sep), "label": lk, "label_frac": round(float(cnt[lk] / sel.sum()), 3), "n": int(sel.sum()),
                    "thickness_mm": round(1e3 * th_col, 2) if th_col is not None else None,
                    "thickness_extent_mm": round(1e3 * th, 2) if th is not None else None, "plan_mm": want, "thick_ok": ok,
                    "thick_ok_extent": bool(sep and want is not None and abs(1e3 * th - want) <= tol_mm)})
    return out


def plan_with_thickness(argv, choices, n_max, seed, margin_mm=16.0):
    """11-26 reports/81: argv 의 '--plan a,b,…' 를 지정 두께 t(choices 에서 seed 로 고름) × n 번으로 바꿈.
    n = min(n_max, (식재료 x 길이 − margin 16 mm(손이 잡을 몸통)) // t) — 길이는 --shape 의 json bounds(없으면 n_max). 다른 인자는 그대로."""
    argv = list(argv)
    t = float(np.random.default_rng(int(seed) + 4242).choice([float(c) for c in str(choices).split(",")]))
    n = int(n_max)
    if "--shape" in argv:
        js = Path(argv[argv.index("--shape") + 1] + ".json")
        if js.exists():
            b = json.loads(js.read_text()).get("bounds")
            if b:
                L = 1e3 * (b[1][0] - b[0][0])
                n = max(1, min(n, int((L - margin_mm) // t)))
    i = argv.index("--plan")
    argv[i + 1] = ",".join(f"{t:g}" for _ in range(n))
    return argv


def run_prefix(argv):
    """og_veg_cut.py 의 준비 부분(MARK 앞)을 새 이름공간에서 돌린다 → 그 이름공간(dict)."""
    src = OGVC.read_text(encoding="utf-8")
    if MARK not in src:
        raise SystemExit(f"{OGVC} 에 '{MARK}' 표시가 없음")
    pre = src.split(MARK)[0]
    g = {"__name__": "og_veg_cut_setup", "__file__": str(OGVC), "__builtins__": __builtins__}
    old = sys.argv
    sys.argv = [str(OGVC)] + list(argv)
    try:
        exec(compile(pre, str(OGVC), "exec"), g)  # noqa: S102 - 우리 저장소의 우리 스크립트
    finally:
        sys.argv = old
    return g


class PlanFollower:
    """우리 칼 경로(1 ms 줄, og_veg_cut.py Qg·PH)를 물리 단계마다 따라간다.

    포인터 s(줄, 실수)를 단계 길이(dt/1 ms 줄)만큼 옮긴다. 칼질(cut) 단계에서 내려가는 중이면 ForceGate.rate() 몫만큼만.
    밀며 썰기 y 는 그 칼질의 cut·bsaw 단계에서 지난 시간으로 사인(og_veg_cut._add_saw 와 같은 진폭·빈도), dwell 동안 0 으로 되돌림.
    멈춤(ForceGate.note)이면 그 칼질 남은 줄은 멈춘 높이 아래로 안 내려감. 단계마다 지난 줄 목록을 돌려줘 부르는 쪽이 줄 사건을 처리한다.
    """

    def __init__(self, g, gate, dt):
        self.g, self.gate, self.dt = g, gate, float(dt)
        self.Q, self.PH, self.n = g["Qg"], g["PH"], int(g["n_steps"])
        self.step_dt = float(g["step_dt"])
        self.amp, self.hz = float(g["args"].saw_amp), float(g["args"].saw_hz)
        self.Qb = np.array(self.Q, float)  # 밀며 썰기를 뺀 경로
        self.SIG = np.full(self.n, -1, np.int64)
        self.DWM = np.full(self.n, -1, np.int64)
        self.nw = {}
        for k, tab in g["SAW_TAB"].items():
            for jj, r in enumerate(tab["ic"]):
                if r < self.n:
                    self.Qb[r, 1] = tab["base_ic"][jj]
                    self.SIG[r] = jj
            for mm, r in enumerate(tab["iw"]):
                if r < self.n:
                    self.Qb[r, 1] = tab["base_iw"][mm]
                    self.DWM[r] = mm
            self.nw[k] = len(tab["iw"])
        self.CUT_OF = np.full(self.n, -1, np.int64)
        for j in range(self.n):
            m = re.match(r"(cut|bsaw|dwell|withdraw)(\d+)$", str(self.PH[j]))
            if m:
                self.CUT_OF[j] = int(m.group(2))
        self.cut_steps = g["cut_steps"]
        self.s = 0.0
        self.t_ms = 0.0
        self.tau = {}  # 칼질 → cut·bsaw 단계에서 지난 시간 [ms]
        self.saw_last = {}
        self.stalled, self.z_stall, self.cuts = {}, {}, {}
        self.q = np.array(self.Q[0], float)
        self.r = 1.0
        self.done = False
        self.track = None  # 11-25: (e_lo, e_hi) [m] — 실제 칼이 명령보다 이만큼 위에 있으면 칼질·바닥 머무름 동안 경로를 늦춤(팔 기다리기)
        self.track_floor = 0.15  # 늦출 때 가장 느린 몫
        self.err_z = 0.0  # 마지막 물리 단계의 실제 칼 높이 − 명령 높이 [m](부르는 쪽이 넣음)
        self.track_hold_ms = 0.0  # 팔을 기다리느라 경로를 세운(늦춘) 시간 합 [ms]

    def _interp(self, s):
        j = min(int(s), self.n - 1)
        j1 = min(j + 1, self.n - 1)
        f = s - j
        return self.Qb[j] + f * (self.Qb[j1] - self.Qb[j])

    def next(self):
        """다음 물리 단계 끝의 칼 목표(우리 좌표 x·y·z·회전), 지난 줄 목록, 이 단계 몫."""
        i0 = min(int(self.s), self.n - 1)
        k = int(self.CUT_OF[i0])
        r = 1.0
        if (self.gate is not None and str(self.PH[i0]).startswith("cut") and not self.stalled.get(k, False)
                and self.Q[min(i0 + 1, self.n - 1), 2] < self.Q[i0, 2]):
            r = self.gate.rate()
        if self.track is not None and str(self.PH[i0]).startswith(("cut", "dwell")) and not self.stalled.get(k, False):
            # 11-25 reports/69: 로봇 팔이 명령을 늦게 따라오는 자세가 있다(시범 ep0: 약 70 ms 늦음 → 칼끝이 도마 3.6 mm 위에서 멈춤).
            # 칼이 명령보다 e_lo 넘게 위에 있으면 경로를 늦추고 e_hi 면 세운다 — 칼질 바닥(dwell)에서 팔이 바닥까지 내려온 뒤에 칼을 듦.
            e_lo, e_hi = self.track
            rt = min(1.0, max(self.track_floor, (e_hi - self.err_z) / max(e_hi - e_lo, 1e-9)))  # 아예 세우지는 않음(11-25 2판 첫 시도: 섬 → 멈춤)
            if rt < 1.0:
                self.track_hold_ms += (1.0 - min(r, rt) / max(r, 1e-9)) * self.dt * 1e3 if r > 0 else 0.0
            r = min(r, rt)
        self.r = r
        rows = self.dt / self.step_dt
        s_new = min(self.s + rows * r, self.n - 1.0)
        crossed = list(range(i0 + 1, int(s_new) + 1))
        q = self._interp(s_new)
        j = min(int(s_new), self.n - 1)
        k2 = int(self.CUT_OF[j])
        if self.amp > 0 and k2 >= 0:
            if self.SIG[j] >= 0:
                self.tau[k2] = self.tau.get(k2, 0.0) + self.dt * 1e3
                self.saw_last[k2] = self.amp * math.sin(2 * math.pi * self.hz * self.tau[k2] * 1e-3)
                q[1] = q[1] + self.saw_last[k2]
            elif self.DWM[j] >= 0 and k2 in self.saw_last:
                q[1] = q[1] + self.saw_last[k2] * (1 - (self.DWM[j] + 1) / max(self.nw.get(k2, 1), 1))
        if self.stalled.get(k2, False):
            q[2] = max(q[2], self.z_stall[k2])
        self.s = s_new
        self.t_ms += self.dt * 1e3
        self.q = q
        if self.s >= self.n - 1.0:
            self.done = True
        return q, crossed, r

    def after(self, crossed, fz, fy):
        """물리 단계 뒤: 칼질 시작·끝, 힘 거르기, 멈춤 판정. 돌려줌: 이번에 시작한 칼질 번호 목록."""
        began = []
        for j in crossed:
            if j in self.cut_steps:
                k = int(self.CUT_OF[j])
                began.append(k)
                if self.gate is not None:
                    self.gate.start_cut(k, self.q[2], int(self.t_ms))
                self.cuts[k] = dict(k=k, t0_ms=self.t_ms, z0=float(self.q[2]), z_min=float(self.q[2]), outcome=None, slow=0,
                                    fz_max=0.0, fz_sum=0.0, n=0, fy_absmax=0.0)
        if self.gate is not None:
            self.gate.push_force(fz)
        i0 = min(int(self.s), self.n - 1)
        k = int(self.CUT_OF[i0])
        c = self.cuts.get(k)
        if c is not None:
            c["z_min"] = min(c["z_min"], float(self.q[2]))
            if str(self.PH[i0]).startswith("cut"):
                c["fz_max"] = max(c["fz_max"], float(fz))
                c["fy_absmax"] = max(c["fy_absmax"], abs(float(fy)))
                c["fz_sum"] += float(fz)
                c["n"] += 1
                c["slow"] += int(self.r < 1.0)
                if self.gate is not None and not self.stalled.get(k, False):
                    stall, why = self.gate.note(float(self.q[2]), int(self.t_ms), self.r < 1.0)
                    if stall:
                        self.stalled[k], self.z_stall[k] = True, float(self.q[2])
                        c.update(outcome="stall", why=why, t_stall_ms=self.t_ms, z_stall=float(self.q[2]))
                        print(f"  힘 한계 {self.gate.cap:g} N: 칼질 {k} 멈춤 — {why} (날 끝 {self.q[2] * 1e3:.2f} mm)", flush=True)
        for j in crossed:  # 칼질 단계 끝
            kk = int(self.CUT_OF[j])
            if kk in self.cuts and str(self.PH[j]).startswith("cut") and (j + 1 >= self.n or self.PH[j + 1] != self.PH[j]):
                self.cuts[kk]["t_cut_end_ms"] = self.t_ms
                if self.cuts[kk]["outcome"] is None:
                    self.cuts[kk]["outcome"] = "cut"
        return began


def rz(yaw):
    c, s_ = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


TIP_SHIFT = 0.0075  # 11-25 reports/77: 손끝 잡는 점 = 원래 잡는 점(손가락 링크 z 45 mm, 손끝보다 9 mm 위 — 11-24 r6b)에서 손끝 쪽으로 7.5 mm


def add_tip_points(robot, shift=TIP_SHIFT):
    """11-25 reports/77(편 집기): 양 손가락에 손끝 가까운 잡는 광선 점을 하나씩 더한다 — 누운 얇은 편(3 mm)은 원래 광선(도마 위 10 mm 아래로 못 감)이
    못 맞히므로. assisted 잡기는 (손가락에 닿은 물체) ∩ (시작점×끝점 광선에 맞은 물체)라 점을 더해도 다른 동작은 그대로. 돌려줌: 더한 두 점(뺄 때 씀)."""
    import torch as th
    from omnigibson.robots.manipulation_robot import GraspingPoint

    s0, e0 = robot._ag_start_points[0], robot._ag_end_points[0]
    ps = GraspingPoint(link_name=s0.link_name, position=th.as_tensor(s0.position) + th.tensor([0.0, 0.0, shift]))
    pe = GraspingPoint(link_name=e0.link_name, position=th.as_tensor(e0.position) + th.tensor([0.0, 0.0, shift]))
    robot._ag_start_points.append(ps)
    robot._ag_end_points.append(pe)
    return ps, pe


def remove_tip_points(robot, added):
    ps, pe = added
    robot._ag_start_points[:] = [g for g in robot._ag_start_points if g is not ps]
    robot._ag_end_points[:] = [g for g in robot._ag_end_points if g is not pe]


def finger_frame(T, robot, arm, idx=None):
    """집게(손끝 틀 기준): 두 손가락 잡는 점의 가운데 c_e, 닫히는 축 y_e, 손끝에서 손가락 쪽으로 뻗는 축 b_e(팀원 grasp_knife 와 같은 잡는 점).
    idx: 잡는 점 목록 중 그 번호만 씀(None = 모두 평균 — 원래 점 하나뿐일 때와 같음. 손끝 점을 더한 뒤에는 0 = 원래, -1 = 손끝)."""
    from scipy.spatial.transform import Rotation

    def ends(points):
        out_ = []
        for gp in points:
            p_, q_ = robot.links[gp.link_name].get_position_orientation()
            out_.append(T.np_(p_) + Rotation.from_quat(T.np_(q_)).as_matrix() @ T.np_(gp.position))
        return np.mean(out_, axis=0)

    sp_, ep_ = robot.assisted_grasp_start_points[arm.arm], robot.assisted_grasp_end_points[arm.arm]
    if idx is not None:
        sp_, ep_ = [sp_[idx]], [ep_[idx]]
    s_, e_ = ends(sp_), ends(ep_)
    E = arm.eef()
    Ri = E[:3, :3].T
    c_e = Ri @ ((s_ + e_) / 2.0 - E[:3, 3])
    y_e = Ri @ ((e_ - s_) / np.linalg.norm(e_ - s_))
    # 뻗는 축 = 손끝 틀 z(손가락 밖으로 — 팀원 손목 카메라 주석과 같음). 11-24 r5: c_e 방향으로 잡았더니 손끝 틀 원점이 잡는 점 가운데와
    # 1.9 mm 밖에 안 떨어져(Franka eef_link 가 손가락 끝) 방향이 엉뚱해 손이 32° 기울어 못 집음.
    b_e = np.array([0.0, 0.0, 1.0])
    y_e = y_e - (y_e @ b_e) * b_e
    y_e /= np.linalg.norm(y_e)
    return c_e, y_e, b_e


def topdown_eef(target, close_dir, ff):
    """손가락 가운데가 target(월드)에 오고, 손가락이 아래(−z)로 뻗고, close_dir(수평) 쪽으로 닫히는 손끝 자세(4×4)."""
    c_e, y_e, b_e = ff
    zw = np.array([0.0, 0.0, -1.0])
    dw = np.asarray(close_dir, float)
    dw = dw - (dw @ zw) * zw
    dw /= np.linalg.norm(dw)
    W = np.stack([np.cross(dw, zw), dw, zw], axis=1)
    Eb = np.stack([np.cross(y_e, b_e), y_e, b_e], axis=1)
    R = W @ Eb.T
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = np.asarray(target, float) - R @ c_e
    return E


def plan_piece_grasp(P_world, names, body, ZB, mode="auto", open_w=0.08, pad_half=0.014, tip_below=0.009, ray_min=0.010,
                     min_h=0.012, inner=0.003, max_w_piece=0.030, upright_w=0.008, lying=None, exclude=()):
    """11-25 reports/72·77: 옮길 조각·닫는 방향·집게 가운데를 고른다(위에서 평행 집게로 집기).

    P_world: 조각마다 지금 월드 점(n,3). 잡는 점 높이 zg = 조각 높이 가운데(도마 위 ray_min 아래로는 안 감 — 손끝이 잡는 점보다
    tip_below 아래라 도마에 닿지 않게). 조각 꼭대기가 zg + 2 mm 보다 낮으면(누운 얇은 편) 잡는 광선이 조각을 못 맞혀 후보에서 뺌.
    닫는 방향 d 를 7.5° 마다 돌려 보며: 조각 폭 W(d 방향 끝–끝)가 열린 폭 − 6 mm 보다 작고, 두 손가락 길(열린 자리 + 12 mm 부터
    조각 끝까지, d 에 수직으로 패드 폭 ±pad_half, 높이 손끝 ~ +5 cm)에 다른 조각 점이 없는 방향을 찾는다(점 수가 적고 W 가 작은 쪽).
    11-25 07:25 2차(양파 seed 1 다시 하기에서 몸통에 기댄 반달 편을 폭 61.7 mm 방향으로 집다가 실패·조각들을 흩뜨림): 패드 폭 ±11 → ±14 mm,
    조각 끝 안쪽 inner(3 mm)부터 막힘으로 셈. 07:50 3차: 몸통 아닌 조각은 **곧게 선 얇은 편**(위에서 본 가장 얇은 폭 ≤ upright_w 8 mm)만,
    닫는 방향은 그 두께 쪽(폭 ≤ 가장 얇은 폭 + 3 mm)만 — 얇은 모서리 둘을 집거나, 기댄 편을 비스듬히·지름 쪽으로 집지 않게. 나머지는 몸통.
    17:30 4차(reports/77 편 집기): lying = 손끝 잡는 광선 높이(도마 위 m, 예 0.0021)를 주면 **누운 편**(높이 2.6 ~ 12 mm)도 후보 — 손끝 가까이 더한
    잡는 광선(손끝보다 1.5 mm 위)을 도마 위 lying 높이에 두고 편 가장자리를 손끝으로 집음(닫는 방향은 아무 쪽, 폭 ≤ 열린 폭 − 6 mm).
    exclude: 후보에서 뺄 조각 번호(이미 접시 위 등).
    mode: auto = 길이 깨끗한 '곧게 선 편' → 길이 깨끗한 '누운 편' → 몸통(막혀도) → 막힌 점이 가장 적은 것 (편을 먼저 — 편 집기 과제),
          body = 몸통, slice = 11-24 규칙(몸통 말고 15 mm 넘게 선 조각 중 가장 큰 것, 가장 얇은 수평 축으로 닫음, 막힘 안 봄).
    반환: (고른 후보 dict — grasp = upright|lying|body, 후보 목록)."""
    cands = []
    for i, P in enumerate(P_world):
        zt, zb = float(P[:, 2].max()), float(P[:, 2].min())
        h = zt - zb
        kind = "body" if i == body else "upright"
        zg = max(0.5 * (zt + zb), ZB + ray_min)
        if i != body and lying is not None and h < min_h and zt >= ZB + lying + 0.0005 and h >= 0.0026:
            kind, zg = "lying", ZB + lying
        c_xy = P[:, :2].mean(axis=0)
        others = np.concatenate([Q for j, Q in enumerate(P_world) if j != i]) if len(P_world) > 1 else np.zeros((0, 3))
        z_lo = (ZB - 0.001) if kind == "lying" else (zg - tip_below - 0.001)
        inz = (others[:, 2] >= z_lo) & (others[:, 2] <= zg + 0.05)
        oth = others[inz]
        best = None
        angs = np.arange(0.0, 180.0, 7.5)
        w_all = []
        for ang in angs:
            r = math.radians(ang)
            s_ = (P[:, :2] - c_xy) @ np.array([math.cos(r), math.sin(r)])
            w_all.append(float(s_.max() - s_.min()))
        # 곧게 선 편: 위에서 본 가장 얇은 폭 ≤ upright_w, 닫는 방향은 그 두께 쪽(+3 mm 안)만(3차 규칙). 누운 편: 아무 쪽(폭 ≤ 열린 폭 − 6 mm).
        if kind == "upright" and min(w_all) > upright_w:
            w_lim = -1.0  # 후보 없음
        elif kind == "upright":
            w_lim = min(max_w_piece, min(w_all) + 0.003)
        else:
            w_lim = open_w - 0.006
        for ang in angs:
            r = math.radians(ang)
            d = np.array([math.cos(r), math.sin(r)])
            e = np.array([-d[1], d[0]])
            s = (P[:, :2] - c_xy) @ d
            smin, smax = float(s.min()), float(s.max())
            W = smax - smin
            if W > open_w - 0.006 or W > w_lim:
                continue
            cc = c_xy + 0.5 * (smin + smax) * d
            so, to = (oth[:, :2] - cc) @ d, (oth[:, :2] - cc) @ e
            half = 0.5 * W
            band = np.abs(to) <= pad_half
            left = (so <= -half + inner) & (so >= -0.5 * open_w - 0.012)
            right = (so >= half - inner) & (so <= 0.5 * open_w + 0.012)
            nb = int(np.sum(band & (left | right)))
            key = (nb, W)
            if best is None or key < best[0]:
                best = (key, ang, d, cc, W, nb)
        ok_ray = bool(zt >= zg + (0.0005 if kind == "lying" else 0.002))
        base = {"k": i, "name": names[i], "body": i == body, "grasp": kind, "height_mm": round(1e3 * h, 1), "ok_ray": ok_ray,
                "excluded": i in exclude}
        if best is None:  # 집게보다 넓거나 조건에 안 맞음
            cands.append(dict(base, width_mm=None, blocked_pts=None))
            continue
        _, ang, d, cc, W, nb = best
        cands.append(dict(base, width_mm=round(1e3 * W, 1), blocked_pts=nb, angle_deg=float(ang), n=int(len(P)),
                          d=np.array([d[0], d[1], 0.0]), c=np.array([cc[0], cc[1], zg]), z_bottom=zb))
    ok = [c for c in cands if c["width_mm"] is not None and c["ok_ray"] and not c["excluded"]]
    if mode == "slice":
        stand = [c for c in cands if not c["body"] and c["height_mm"] >= 15.0 and not c["excluded"]]
        pick = max(stand, key=lambda c: c["n"] if "n" in c else 0) if stand else next(c for c in cands if c["body"])
        if pick.get("width_mm") is None:
            raise RuntimeError(f"고른 조각 {pick['name']} 이 집게보다 넓음")
        P = P_world[pick["k"]]
        w_, v_ = np.linalg.eigh(np.cov((P[:, :2] - P[:, :2].mean(axis=0)).T))
        pick = dict(pick, grasp="upright", d=np.array([v_[0, 0], v_[1, 0], 0.0]),  # 11-24 그대로: 가장 얇은 수평 축, 가운데 = 무게중심
                    c=np.array([*P[:, :2].mean(axis=0), max(0.5 * (P[:, 2].max() + P[:, 2].min()), ZB + ray_min)]))
        return pick, cands
    if mode == "body":
        b = [c for c in ok if c["body"]]
        if b:
            return b[0], cands
    else:
        for kind in ("upright", "lying"):
            clear = [c for c in ok if c["blocked_pts"] == 0 and c["grasp"] == kind and c["height_mm"] >= (1e3 * min_h if kind == "upright" else 0.0)]
            if clear:
                return max(clear, key=lambda c: c["n"]), cands
        b = [c for c in ok if c["body"] and c["blocked_pts"] == 0]
        if b:
            return b[0], cands
    if not ok:
        raise RuntimeError("잡을 수 있는 조각이 없음(모두 집게보다 넓거나 누워 있음)")
    b = [c for c in ok if c["body"]]  # 깨끗한 후보가 없으면 몸통 먼저(다시 하기: 막힘 13점 몸통도 다른 조각 0 mm 밀고 잡힘)
    if b:
        return b[0], cands
    return min(ok, key=lambda c: c["blocked_pts"]), cands


def move_hand(T, run, E1, gripper, speed=0.15, ang_speed=0.8, settle=0.3, phase="reach"):
    """손끝을 E1 으로(최소 저크), 그 뒤 settle 초 머묾(팀원 move_knife 와 같은 식, 칼 대신 손끝)."""
    run.phase = phase
    E0 = run.arm.eef()
    dist = float(np.linalg.norm(E1[:3, 3] - E0[:3, 3]))
    ang = math.radians(T.rotation_deg(E0[:3, :3], E1[:3, :3]))
    n = int(math.ceil(max(0.4, dist / speed, ang / ang_speed) / run.dt))
    for i in range(1, n + 1):
        run.step(T.interpolate(E0, E1, T.min_jerk(i / n)), gripper)
    for _ in range(int(round(settle / run.dt))):
        run.step(E1, gripper)


def grasp_knife_geo(T, run, knife, robot, arm, geometry, canonical=True):
    """팀원 tofu_og_cut.grasp_knife 와 같되 칼 손잡이 가운데를 우리 칼 치수로(handle_centre(geometry)).

    11-24 r3 진단: 팀원 함수는 handle_centre()(기본 = 팀원 셰프 칼, 칼날 가운데에서 167 mm 뒤)를 써서, 칼날이 짧은 우리 칼(80 mm)은
    손잡이 끝 5 mm 너머를 쥠 → 두 손가락은 닿지만 손끝 사이 광선이 손잡이를 못 맞혀 assisted 잡기가 안 걸림.
    """
    import torch as th
    from scipy.spatial.transform import Rotation
    from simfoundry.cutting.cosim import matrix_to_quat_xyzw
    from simfoundry.cutting.og_cosim import handle_centre

    run.phase = "grasp"
    if canonical:
        # 11-25 reports/73: 칼은 '지금 손 방향' 기준으로 손에 들어가므로(아래 R), 쥐기 전 손 자세가 칼–손 관계를 정한다.
        #   2판: 첫 에피소드는 로봇 처음 자세(손 25° 기욺) → 칼이 손 안에서 25° 기울어 썰 때 손목이 비틀림(j5 −1.05·j6 3.0 rad,
        #   팔이 칼을 늦게 따라감), 다음 에피소드는 옮기기 끝 손 방향 그대로 → 오이 seed 1 은 칼이 앞뒤로 뒤집혀 손목 카메라에 식재료가 안 보임.
        #   → 늘 같은 자세(손끝 아래, 손 x = 로봇 받침에서 멀어지는 수평 방향)로 옮긴 뒤 쥠: 칼–손 = diag(1, −1, −1)(2판 당근 seed 1 과 같음).
        E0 = arm.eef()
        xh = E0[:3, 3] - arm.base[:3, 3]
        xh[2] = 0.0
        xh /= np.linalg.norm(xh)
        zh = np.array([0.0, 0.0, -1.0])
        Ec = np.eye(4)
        Ec[:3, :3] = np.stack([xh, np.cross(zh, xh), zh], axis=1)
        Ec[:3, 3] = E0[:3, 3]
        move_hand(T, run, Ec, T.OPEN, speed=0.15, ang_speed=0.8, settle=0.3, phase="grasp")
    hold = arm.eef()
    for _ in range(30):
        run.step(hold, T.OPEN)

    def ends(points):
        out_ = []
        for gp in points:
            p_, q_ = robot.links[gp.link_name].get_position_orientation()
            out_.append(T.np_(p_) + Rotation.from_quat(T.np_(q_)).as_matrix() @ T.np_(gp.position))
        return np.mean(out_, axis=0)

    # [:1] = 원래 잡는 점만(11-25: 손끝 점을 더했을 때도 칼 쥐는 자리가 그대로이게)
    s_, e_ = ends(robot.assisted_grasp_start_points[arm.arm][:1]), ends(robot.assisted_grasp_end_points[arm.arm][:1])
    centre, y = (s_ + e_) / 2.0, (e_ - s_) / np.linalg.norm(e_ - s_)
    z = np.array([0.0, 0.0, 1.0]) - y[2] * y
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    if x[:2] @ (centre - arm.base[:3, 3])[:2] < 0.0:
        x, y = -x, -y
    R = np.stack([x, y, z], axis=1)
    placed = centre - R @ handle_centre(geometry)
    knife.root_link.disable_gravity()
    knife.set_position_orientation(position=th.tensor(placed, dtype=th.float32), orientation=th.tensor(matrix_to_quat_xyzw(R), dtype=th.float32))
    knife.set_linear_velocity(th.zeros(3))
    knife.set_angular_velocity(th.zeros(3))
    attached = None
    for i in range(120):
        run.step(hold, T.CLOSE)
        if robot.is_grasping(arm.arm, candidate_obj=knife) == run.grasp_true:
            attached = i + 1
            break
    for _ in range(30):
        run.step(hold, T.CLOSE)
    K = run.knife_pose()
    E = arm.eef()
    return {"attached_after_steps": attached, "held": robot.is_grasping(arm.arm, candidate_obj=knife) == run.grasp_true,
            "moved_while_closing_mm": 1e3 * float(np.linalg.norm(K[:3, 3] - placed)),
            "turned_while_closing_deg": T.rotation_deg(K[:3, :3], R), "fingers_m": arm.fingers().tolist(),
            "knife_in_eef": (np.linalg.inv(E) @ K).tolist(), "handle_centre_m": handle_centre(geometry).tolist()}


def main():
    a, food_argv = parse()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()

    fast3_note = None
    if a.fast3:  # 11-25 reports/71: 솔버를 빠른3 복사본으로(도장이 지금 작업 트리와 같을 때만)
        import hashlib
        fp = Path(a.fast3_path)
        stamp = fp / "FAST3_BASE.json"
        if not stamp.exists():
            fast3_note = f"{fp} 없음 → 원래 솔버(만들기: bash patches/build_datagen_fast3.sh)"
        else:
            st = json.loads(stamp.read_text())
            team = Path(st.get("base", ""))
            now = {str(f.relative_to(team)): hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in sorted((team / "simfoundry").rglob("*.py")) if "__pycache__" not in f.parts}
            if st.get("base_sha256") != now:
                fast3_note = f"작업 트리 {team}/simfoundry 가 빠른 복사본을 만든 뒤 바뀜 → 원래 솔버(다시: bash patches/build_datagen_fast3.sh)"
            else:
                sys.path.insert(0, str(fp))
        if fast3_note:
            a.fast3 = False
            print(f"[food_robot] [경고] 빠른3 못 씀: {fast3_note}", flush=True)

    from simfoundry import configure_omnigibson_data_path

    configure_omnigibson_data_path(force=False)
    import omnigibson as og
    from omnigibson.macros import gm

    from simfoundry.utils.og_utils import apply_teleop_omnigibson_macros

    apply_teleop_omnigibson_macros(enable_tr=False)
    gm.ENABLE_OBJECT_STATES = False
    dt = 1.0 / a.physics_hz
    og.launch(physics_dt=dt, rendering_dt=dt, sim_step_dt=dt, device="cpu")
    import torch as th
    import warp as wp  # noqa: F401 - og.launch 뒤: Isaac Sim 의 Warp
    from omnigibson.controllers import IsGraspingState

    import simfoundry
    sys.path.insert(0, str(Path(simfoundry.__file__).resolve().parents[1] / "scripts" / "cutting"))
    import tofu_og_cut as T  # noqa: E402 - 팀원 로봇·칼 도구
    import tofu_policy_episodes as P  # noqa: E402 - 팀원 기록·힘 상한 결합기
    from simfoundry.cutting.cosim import KnifeCoupling, matrix_to_quat_xyzw
    from simfoundry.cutting.og_cosim import FoodPoints, PhysicsStepCoupler, handle_centre, write_knife_usd
    sys.path.insert(0, str(HERE))
    from force_cap import ForceGate

    print(f"[food_robot] 팀원 코드: {Path(T.__file__).resolve()}, 솔버: {Path(simfoundry.__file__).resolve().parent}"
          f"{' (빠른3)' if a.fast3 else ''}", flush=True)

    class PlanCoupling(KnifeCoupling):
        """칼질 시작을 우리 경로(cut 단계 첫 줄)로 알려 준다(팀원 자동 규칙 대신 — 순간 이동 쌓기 뒤에도 맞게)."""
        pending = False

        def step(self, dt_, link_pos, link_quat_xyzw, begin_cut=None):
            b = bool(self.pending)
            self.pending = False
            return super().step(dt_, link_pos, link_quat_xyzw, begin_cut=b)

    Coupler = P.capped_coupler(PhysicsStepCoupler, *a.wrench_cap)
    rng0 = np.random.default_rng(a.seed0)

    def setup(seed):
        argv = list(food_argv)
        wet = None
        if a.wet_lognormal:
            m, sg = a.wet_lognormal
            wet = float(math.exp(math.log(m) + sg * np.random.default_rng(seed + 7919).normal()))
            argv += ["--wet", f"{wet:.4f}"]
        if a.thick_choices:  # 11-26 reports/81: 에피소드마다 지정 두께를 하나 뽑아 --plan 을 바꿈(과제 문장도 그 두께)
            argv = plan_with_thickness(argv, a.thick_choices, a.thick_n, seed)
        g = run_prefix(argv + ["--no_og", "--fast3" if a.fast3 else "--no_fast3", "--out", str(out / f"ep_{seed:05d}_phys")])
        return g, wet

    # ---- 장면(처음 한 번): 첫 에피소드 준비로 칼·탁자·도마 치수를 안다
    g0, wet0 = setup(a.seed0)
    TABLE_H, BOARD_T, ZB = g0["TABLE_H"], g0["BOARD_T"], g0["ZB"]
    knife_geo = g0["knife_geo"]
    knife_usd = write_knife_usd(out / "knife" / "knife.usd", knife_geo)
    Wm0 = np.eye(4)
    Wm0[:3, 3] = g0["W_OFF"]
    hc = np.append(handle_centre(knife_geo), 1.0)
    handles = np.array([(T.knife_world(g0["knife_pose_our"](q), Wm0) @ hc)[:3] for q in g0["Qg"][::50]])
    hm = handles.mean(axis=0)
    # 11-24 r1: 바닥(z 0)에 세우면 처음 팔 자세가 통짜 탁자(높이 0.76 m) 안으로 들어가 칼이 튕겨 나감(손가락 사이에 놓은 칼이 207 mm 밀림)
    #   → 탁자 높이 바로 아래 받침대 위에(고정 받침 로봇, 탁자 위 작업에 흔한 놓임). 받침대는 보이는 상자로 둠.
    ROBOT_Z = TABLE_H - a.robot_below_table
    robot_pos = np.array([hm[0], hm[1] + a.standoff, ROBOT_Z])
    robot_yaw = -0.5 * math.pi  # 팀원 'behind': 칼 끝이 로봇 반대쪽(-y)
    PLATE_XY = (-0.25, 0.22)  # 과제 ③ 접시(탁자 위, 도마 밖, 로봇 쪽)
    if a.plate_xy is not None:
        PLATE_XY = tuple(a.plate_xy)
    elif a.unstick == "wipe":  # 11-25 밤 reports/78: 접시를 도마 오른쪽(+x) 앞에 — 첫 팔은 손잡이 쪽(+y, 뒤)에서, 도움 팔은 칼끝 쪽(−y, 앞)에서
        PLATE_XY = (0.27, 0.0)
    elif a.unstick == "scrape":  # 30 cm 네모 접시가 도마(x ≥ −0.17)와 안 겹치게
        PLATE_XY = (-0.27, 0.22)
    use_tray = a.unstick == "scrape"
    use_bowl = a.bowl is not None and not use_tray
    PLATE_TOP = TABLE_H + (0.006 if (use_tray or use_bowl) else 0.012)  # 접시 바닥 윗면
    if a.task_kind == "move":
        a.to_rigid = True
    objects = [
        dict(type="PrimitiveObject", name="table", primitive_type="Cube", fixed_base=True, rgba=[0.86, 0.85, 0.82, 1.0],
             scale=[0.9, 0.7, TABLE_H], position=[0.0, 0.0, TABLE_H / 2]),
        dict(type="PrimitiveObject", name="robot_stand", primitive_type="Cube", fixed_base=True, rgba=[0.35, 0.35, 0.38, 1.0],
             scale=[0.22, 0.22, ROBOT_Z - 0.005], position=[robot_pos[0], robot_pos[1], (ROBOT_Z - 0.005) / 2]),
        dict(type="PrimitiveObject", name="board", primitive_type="Cube", fixed_base=True, rgba=[0.50, 0.33, 0.18, 1.0],
             scale=[0.34, 0.24, BOARD_T], position=[0.0, 0.0, TABLE_H + BOARD_T / 2]),
        *([dict(type="PrimitiveObject", name="plate", primitive_type="Cylinder", fixed_base=True, rgba=[0.95, 0.95, 0.93, 1.0],
                scale=[0.16, 0.16, 0.012], position=[PLATE_XY[0], PLATE_XY[1], TABLE_H + 0.006])]
          if a.task_kind in ("move", "move_replay", "unstick_replay") and not use_tray and not use_bowl else []),
        *([dict(type="USDObject", name="plate", category="bowl", fixed_base=True,  # 11-26 reports/79: 둥근 그릇
                usd_path=__import__("food_handoff").write_bowl_usd(out / "bowl" / "bowl.usd", "bowl", a.bowl[0], a.bowl[1]),
                position=[PLATE_XY[0], PLATE_XY[1], TABLE_H])] if use_bowl and a.task_kind in ("move", "move_replay", "unstick_replay") else []),
        *([dict(type="USDObject", name="plate", category="plate", fixed_base=True,  # 11-25 밤 reports/78: 긁기용 곧은 벽 네모 접시
                usd_path=__import__("food_handoff").write_tray_usd(out / "tray" / "tray.usd", "tray", a.tray[0], a.tray[1], a.tray[2], end_h=a.tray[3],
                                                                   gap_py=a.tray_gap),
                position=[PLATE_XY[0], PLATE_XY[1], TABLE_H])] if use_tray and a.task_kind in ("move", "move_replay", "unstick_replay") else []),
        dict(type="USDObject", name=T.KNIFE, usd_path=str(knife_usd), category="knife", fixed_base=False,
             position=[0.30, -0.28, TABLE_H + 0.02], orientation=[0.7071068, 0.0, 0.0, 0.7071068]),
    ]
    robots = [dict(type="FrankaPanda", name="robot0", obs_modalities=["proprio"], action_normalize=False, self_collisions=False,
                   grasping_mode="assisted", end_effector="gripper", position=robot_pos.tolist(),
                   orientation=[0.0, 0.0, math.sin(0.5 * robot_yaw), math.cos(0.5 * robot_yaw)],
                   controller_config={"arm_0": T.arm_controller_config("ik", 150.0, 10.0),
                                      "gripper_0": {"name": "MultiFingerGripperController", "command_input_limits": [0.0, 1.0],
                                                    "mode": "smooth"}})]
    HELPER_POS = None
    if a.unstick == "wipe" and a.task_kind in ("move", "unstick_replay"):  # 11-25 밤 reports/78: 반대쪽 손 = 두 번째 Franka(도움 팔), −x(접시) 쪽을 봄
        HELPER_POS = np.array([a.helper_xy[0], a.helper_xy[1], ROBOT_Z])
        objects.append(dict(type="PrimitiveObject", name="helper_stand", primitive_type="Cube", fixed_base=True, rgba=[0.35, 0.35, 0.38, 1.0],
                            scale=[0.22, 0.22, ROBOT_Z - 0.005], position=[HELPER_POS[0], HELPER_POS[1], (ROBOT_Z - 0.005) / 2]))
        yaw_h = math.atan2(PLATE_XY[1] - HELPER_POS[1], PLATE_XY[0] - HELPER_POS[0])  # 접시 쪽을 봄
        robots.append(dict(type="FrankaPanda", name="helper", obs_modalities=["proprio"], action_normalize=False, self_collisions=False,
                           grasping_mode="physical", end_effector="gripper", position=HELPER_POS.tolist(),
                           orientation=[0.0, 0.0, math.sin(0.5 * yaw_h), math.cos(0.5 * yaw_h)],
                           controller_config={"arm_0": T.arm_controller_config("ik", 150.0, 10.0),
                                              "gripper_0": {"name": "MultiFingerGripperController", "command_input_limits": [0.0, 1.0],
                                                            "mode": "smooth"}}))
    centre_w = g0["W_OFF"] + np.array([0.0, 0.0, 0.5 * g0["ZT"]])
    az, el = math.radians(-50.0), math.radians(32.0)  # 팀원 front 카메라와 같은 자리 잡기(식재료 가운데에서 0.45 m)
    cam_d = 0.45
    if a.task_kind == "place":  # 탁자 위 시작 자리(x 0.20~0.28, y 0.14~0.22)와 도마가 함께 보이게: 그 가운데를 0.62 m 에서
        centre_w = np.array([0.12, 0.09, g0["ZB"] + 0.02])
        cam_d = 0.62
    elif a.task_kind in ("move", "move_replay", "unstick_replay"):  # 도마와 접시가 함께 보이게
        centre_w = np.array([0.5 * PLATE_XY[0] + 0.005, 0.10, g0["ZB"] + 0.02])  # 접시 −0.25 이면 예전과 같은 −0.12
        cam_d = 0.62
    eye = centre_w + cam_d * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    sensors = [{"sensor_type": "VisionSensor", "name": n, "modalities": ["rgb"],
                "sensor_kwargs": {"image_height": a.img, "image_width": a.img, "clipping_range": [0.01, 20.0]},
                "position": [float(v) for v in eye], "orientation": [float(v) for v in T.look_at(eye, centre_w)]}
               for n in ("front", "wrist")]
    if a.overview:
        eye_o = centre_w + np.array([0.55, -0.75, 0.55])
        sensors.append({"sensor_type": "VisionSensor", "name": "overview", "modalities": ["rgb"],
                        "sensor_kwargs": {"image_height": a.overview[1], "image_width": a.overview[0], "clipping_range": [0.01, 20.0]},
                        "position": [float(v) for v in eye_o],
                        "orientation": [float(v) for v in T.look_at(eye_o, centre_w + np.array([0.0, 0.25, 0.1]))]})
    hz = a.physics_hz
    env = og.Environment(configs=dict(
        env=dict(action_frequency=hz, rendering_frequency=hz, physics_frequency=hz, external_sensors=sensors),
        scene=dict(type="Scene", use_floor_plane=True, floor_plane_visible=True, use_skybox=True),
        robots=robots, objects=objects))
    env.reset()
    cams = {n: env.external_sensors[n] for n in ("front", "wrist") + (("overview",) if a.overview else ())}
    robot = next(r_ for r_ in env.robots if r_.name == "robot0")  # 11-25 밤: 이름으로(도움 팔 'helper' 가 있으면 env.robots 순서가 이름순이라 [0] 이 도움 팔이었음)
    knife = env.scene.object_registry("name", T.KNIFE)
    arm = T.Arm(robot)
    wrist = np.eye(4)
    wrist[:3, :3] = np.stack([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], axis=1)
    wrist[:3, 3] = a.wrist
    class ViewRun(P.EpisodeRun):
        """팀원 기록기 + 메쉬 식재료(11-25 B5): 화면을 찍기 직전(틱·넓은 카메라)에 메쉬를 지금 입자·강체 자리로 옮긴다."""
        food_view = None
        t_cap = t_wide = 0.0  # 11-25 reports/71: 시간 나눠 재기(학습 화면 찍기·넓은 카메라 프레임, 메쉬 새로 고침 포함)
        n_cap = n_wide = 0
        post_step = None  # 11-25 reports/77: 물리 단계마다 부를 것(칼에 붙인 편의 접착 하중 감시)
        helper_arm = None  # 11-25 밤 reports/78: 도움 팔(두 번째 Franka) — 단계마다 helper_cmd(손 자세, 집게)를 줌(안 주면 팔이 처짐)
        helper_cmd = None

        def step(self, *args_, **kw_):
            # 11-26 reports/81: 이번 단계의 집게 명령(팀원 T.OPEN 1 / T.CLOSE 0) — 틱 기록 'grip' 으로(학습 행동에 집게 칸을 넣을 때 씀)
            self.grip_now = float(args_[1]) if len(args_) > 1 else float(kw_.get("gripper", float("nan")))
            if self.helper_arm is not None and self.helper_cmd is not None:
                self.helper_arm.robot.apply_action(self.helper_arm.action(*self.helper_cmd))
            K_ = super().step(*args_, **kw_)
            if self.post_step is not None:
                mon_ = getattr(self.post_step, "__self__", None)
                if mon_ is not None and hasattr(mon_, "phase"):
                    mon_.phase = self.phase
                self.post_step()
            return K_

        def capture(self):
            t0_ = time.perf_counter()
            try:
                return self._capture()
            finally:
                self.t_cap += time.perf_counter() - t0_
                self.n_cap += 1

        def _capture(self):
            if self.food_view is not None:
                self.food_view.refresh()
            if self.rec is not None and len(self.rec) == 0:
                # 11-25: 에피소드 첫 틱 — 손목 카메라를 지금 손 자리로 옮긴 뒤 몇 번 더 그림. 안 하면 첫 wrist 사진이 앞 자리(멀리 있던 카메라)를 찍었다(m1·m6·m7, 팀원 기록기도 같음)
                C = self.arm.eef() @ self.wrist_offset
                self.cams["wrist"].set_position_orientation(position=th.tensor(C[:3, 3], dtype=th.float32),
                                                            orientation=th.tensor(matrix_to_quat_xyzw(C[:3, :3]), dtype=th.float32))
                for _ in range(15):  # 11-25: 4 번은 자리는 맞아도 흐릿했음(2판 첫 손목 사진 — 화면 누적 덜 됨) → 팀원 warmup_renders 와 같은 15 번
                    self.og.sim.render()
            if a.look_preview > 0 and self.rec is not None and len(self.rec) == 0:
                look_preview_and_exit(self)
            r_ = super().capture()
            r_["grip"] = getattr(self, "grip_now", float("nan"))
            if self.helper_arm is not None and self.helper_cmd is not None:  # 도움 팔 상태·명령(받침 틀: 자리 3 + 회전 6, 집게)
                ha = self.helper_arm
                Eh, Ch = ha.base_inv @ ha.eef(), ha.base_inv @ self.helper_cmd[0]
                r_["helper_state"] = np.concatenate([ha.joints(), ha.fingers(), Eh[:3, 3], matrix_to_quat_xyzw(Eh[:3, :3])]).astype(np.float32)
                r_["helper_cmd"] = np.concatenate([Ch[:3, 3], Ch[:3, :3][:, :2].T.reshape(-1), [self.helper_cmd[1]]]).astype(np.float32)
            return r_

        def wide_frame(self):
            t0_ = time.perf_counter()
            try:
                if self.food_view is not None:
                    self.food_view.refresh()
                return super().wide_frame()
            finally:
                self.t_wide += time.perf_counter() - t0_
                self.n_wide += 1

    import food_look as FL  # 11-26 reports/81: 화면 겉모습(물리 그대로)
    LOOK = FL.SceneLook(og, env, cams, table_h=TABLE_H, table_xy=(0.0, 0.0), table_size=(0.9, 0.7), board_xy=(0.0, 0.0),
                        board_size=(0.34, 0.24, BOARD_T), board_z0=TABLE_H, front=("front", eye, centre_w), look_at=T.look_at)

    def look_for(seed_):
        """에피소드 seed → 이번 겉모습(dict 또는 None)을 입히고 돌려줌."""
        lk = FL.sample_look(a.look, a.look_seed if a.look_seed is not None else seed_)
        if lk is not None or LOOK.current is not None:
            LOOK.apply(lk)
            for _ in range(8):  # 무늬·돔 그림 불러오기 기다림
                og.sim.render()
        return lk

    def look_preview_and_exit(run_):
        """첫 틱에서 겉모습 여러 벌로 찍어 저장하고 프로세스를 끝냄(시험용)."""
        import imageio.v2 as imageio

        pdir = out / "look_preview"
        pdir.mkdir(parents=True, exist_ok=True)
        C_ = run_.arm.eef() @ run_.wrist_offset
        run_.cams["wrist"].set_position_orientation(position=th.tensor(C_[:3, 3], dtype=th.float32),
                                                    orientation=th.tensor(matrix_to_quat_xyzw(C_[:3, :3]), dtype=th.float32))
        looks = [("base", None), ("real", FL.sample_look("real", 0))] + [(f"random{k:02d}", FL.sample_look("random", k)) for k in range(a.look_preview)]
        for nm, lk in looks:
            LOOK.apply(lk)
            for _ in range(40):
                og.sim.render()
            for cn, cam_ in run_.cams.items():
                im = T.np_(cam_.get_obs()[0]["rgb"])[:, :, :3].astype(np.uint8)
                imageio.imwrite(pdir / f"{nm}_{cn}.png", im)
            (pdir / f"{nm}.json").write_text(json.dumps(lk, ensure_ascii=False, indent=1))
            print(f"[food_robot] 겉모습 미리 보기 {nm} 저장", flush=True)
        print(f"[food_robot] 겉모습 미리 보기 끝 → {pdir}", flush=True)
        os._exit(0)

    import food_mesh_view as FMV
    run = ViewRun(og, env, arm, knife, None, cams, out, a.policy_hz, set(), wrist_offset=wrist)
    run.grasp_true = IsGraspingState.TRUE
    helper = None
    if HELPER_POS is not None:  # 11-25 밤 reports/78: 도움 팔 — 지금 손 자세를 붙잡고(닫은 집게) 시작
        helper = next(r_ for r_ in env.robots if r_.name == "helper")
        # 처음 자세(팔을 앞으로 뻗음, 손이 첫 로봇 앞 (0.03, 0.36, 1.12))가 칼 옮기는 길을 막음(11-25 밤 w1) → 받침 쪽으로 접어 둠
        q_t = T.np_(helper.get_joint_positions()).copy()
        q_t[:7] = [0.0, -1.0, 0.0, -2.6, 0.0, 1.7, 0.785]
        helper.set_joint_positions(th.tensor(q_t, dtype=th.float32))
        helper.keep_still()
        run.helper_arm = T.Arm(helper)
        og.sim.step()
        run.helper_cmd = (run.helper_arm.eef(), T.CLOSE)
        print(f"[food_robot] 도움 팔: 받침 {np.round(HELPER_POS, 3).tolist()}, 손 {np.round(run.helper_arm.eef()[:3, 3], 3).tolist()}", flush=True)
    mesh_cache = out / "mesh_cache"

    def new_view(name, mdir):
        """메쉬 식재료를 장면에 만든다(앞 에피소드 것은 숨김)."""
        if run.food_view is not None:
            run.food_view.hide()
        v = FMV.FoodMesh(og.sim.stage, f"/World/{name}", mdir, residual=a.mesh_residual)
        run.food_view = v
        return v
    def do_move(objs, recs, seed, e, veg, tag=""):
        """과제 ③(11-24 C, 11-25 함수로): 강체 조각 하나를 위에서 집어 접시로. 칼은 이미 치워져 있어야 함. → 결과 dict(없으면 None)."""
        from scipy.spatial.transform import Rotation
        ff = finger_frame(T, robot, arm)
        body = max(range(len(recs)), key=lambda i_: recs[i_]["n"])
        info_p = []
        for i_, o_ in enumerate(objs):
            pp_, pq_ = (T.np_(v) for v in o_.get_position_orientation())
            lo_ = recs[i_]["local"].astype(float) @ Rotation.from_quat(pq_).as_matrix().T
            info_p.append((pp_, pq_, lo_))
        P_world = [pp_ + lo_ for pp_, _, lo_ in info_p]
        # 11-25 reports/77: 이미 접시 위에 있는 조각(칼에서 털어 낸 편)은 빼고, 누운 편도 손끝으로 집을 수 있게(--tip_pinch)
        on_plate0 = {i_ for i_, P_ in enumerate(P_world) if in_plate(P_) and P_[:, 2].min() > TABLE_H + 0.005}  # 11-26: 네모 그릇도(in_plate)
        pick = None
        if a.move_n > 1:  # 11-26 reports/79: 여러 조각을 옮길 때는 썬 조각(편·채)을 먼저, 몸통은 그것이 없을 때만
            try:
                pick, cands = plan_piece_grasp(P_world, [r_["name"] for r_ in recs], body, ZB, mode=a.move_pick,
                                               lying=(a.tip_ray_h if a.tip_pinch else None), exclude=on_plate0 | {body})
            except RuntimeError:
                pick = None
        if pick is None:
            try:
                pick, cands = plan_piece_grasp(P_world, [r_["name"] for r_ in recs], body, ZB, mode=a.move_pick,
                                               lying=(a.tip_ray_h if a.tip_pinch else None), exclude=on_plate0)
            except RuntimeError as err:
                print(f"[food_robot] move seed {seed}: {err}", flush=True)
                return None
        k_p = pick["k"]
        po = objs[k_p]
        pp, pq, loc = info_p[k_p]
        close_d, grasp_c = pick["d"], pick["c"]
        tip_added = None
        if pick["grasp"] == "lying":  # 손끝 잡는 점을 더하고 그 점을 기준으로 손 자세를 잡음(손끝이 도마 위 약 0.6 mm)
            tip_added = add_tip_points(robot)
            ff = finger_frame(T, robot, arm, idx=-1)
        rngm = np.random.default_rng(seed + 104729)
        tgt = np.array([PLATE_XY[0] + rngm.uniform(-0.02, 0.02), PLATE_XY[1] + rngm.uniform(-0.02, 0.02), PLATE_TOP])
        move_hand(T, run, topdown_eef(grasp_c + np.array([0.0, 0.0, 0.12]), close_d, ff), T.OPEN, speed=0.25, settle=0.3, phase="home")
        mdir = out / "move"
        mdir.mkdir(exist_ok=True)
        for _ in range(15):  # DLSS 가 앞 장면을 섞어 그리지 않게(팀원 warmup_renders)
            og.sim.render()
        run.rec, run.cmd_steps, run.rows = [], [], []
        P_before = [p_.copy() for p_ in P_world]
        move_hand(T, run, topdown_eef(grasp_c + np.array([0.0, 0.0, 0.03]), close_d, ff), T.OPEN, speed=0.12, settle=0.1, phase="approach")
        move_hand(T, run, topdown_eef(grasp_c, close_d, ff), T.OPEN, speed=0.04, settle=0.2, phase="descend")
        for _ in range(int(round(0.6 / dt))):
            run.phase = "close"
            run.step(topdown_eef(grasp_c, close_d, ff), T.CLOSE)
        held0 = robot.is_grasping(arm.arm, candidate_obj=po) == run.grasp_true
        # 집는 동안 다른 조각이 밀린 거리(손가락이 이웃을 건드렸나)
        others_moved = []
        for i_, o_ in enumerate(objs):
            if i_ == k_p:
                continue
            q_ = T.np_(o_.get_position_orientation()[0])
            others_moved.append(round(1e3 * float(np.linalg.norm(q_ - info_p[i_][0])), 2))
        move_hand(T, run, topdown_eef(grasp_c + np.array([0.0, 0.0, 0.10]), close_d, ff), T.CLOSE, speed=0.10, settle=0.1, phase="lift")
        drop = tgt + np.array([0.0, 0.0, (grasp_c[2] - pick["z_bottom"]) + 0.008])
        move_hand(T, run, topdown_eef(drop + np.array([0.0, 0.0, 0.10]), close_d, ff), T.CLOSE, speed=0.15, settle=0.2, phase="carry")
        move_hand(T, run, topdown_eef(drop, close_d, ff), T.CLOSE, speed=0.05, settle=0.2, phase="lower")
        for _ in range(int(round(0.5 / dt))):
            run.phase = "release"
            run.step(topdown_eef(drop, close_d, ff), T.OPEN)
        move_hand(T, run, topdown_eef(drop + np.array([0.0, 0.0, 0.12]), close_d, ff), T.OPEN, speed=0.10, settle=0.5, phase="retreat")
        pe = T.np_(po.get_position_orientation()[0])
        held_end = robot.is_grasping(arm.arm, candidate_obj=po) == run.grasp_true
        d_plate = 1e3 * float(np.linalg.norm(pe[:2] - np.array(PLATE_XY)))
        on_plate = bool(d_plate < 75.0 and pe[2] > PLATE_TOP - 0.002)
        others_on_plate = [recs[i_]["name"] for i_, o_ in enumerate(objs) if i_ != k_p and i_ not in on_plate0  # 털기로 먼저 올라간 편은 뺌
                           and np.linalg.norm(T.np_(o_.get_position_orientation()[0])[:2] - np.array(PLATE_XY)) < 0.075]
        mo = {"piece": recs[k_p]["name"], "is_body": bool(k_p == body), "piece_height_mm": pick["height_mm"],
              "piece_n": int(recs[k_p]["n"]), "piece_mass_g": round(recs[k_p]["mass_g"], 2),
              "piece_extent_mm": recs[k_p]["extent_mm"], "pick_mode": a.move_pick, "grip_width_mm": pick.get("width_mm"),
              "close_angle_deg": pick.get("angle_deg"), "blocked_pts": pick.get("blocked_pts"),
              "candidates": [{k_: v for k_, v in c_.items() if k_ not in ("d", "c", "z_bottom")} for c_ in cands],
              "held_after_close": bool(held0), "held_at_end": bool(held_end), "others_moved_at_close_mm": others_moved,
              "others_on_plate": others_on_plate, "dist_from_plate_centre_mm": round(d_plate, 1), "on_plate": on_plate,
              "ticks": len(run.rec), "pos_err_mm_max": round(float(max((r_.get("pos_err_mm", 0.0) for r_ in run.rows), default=0.0)), 3)}
        mo["others_moved_max_mm"] = max(others_moved, default=0.0)
        mo["success"] = bool(held0 and not held_end and on_plate and not others_on_plate)
        mo["clean"] = bool(mo["success"] and mo["others_moved_max_mm"] < 5.0)  # 11-25: 집으며 다른 조각을 5 mm 넘게 밀지 않음
        mo["food_view"] = run.food_view.stats() if run.food_view is not None else a.food_view
        mo["grasp"] = pick["grasp"]
        mo["excluded_on_plate"] = [recs[i_]["name"] for i_ in sorted(on_plate0)]
        if tip_added is not None:
            remove_tip_points(robot, tip_added)
        Rm = run.rec
        _what = "the rest of the" if k_p == body else "a slice of"
        minfo = {"seed": seed, "look": LOOK.current, "mode": "expert_move", "task": a.task or f"move {_what} {veg.replace('_', ' ')} to the plate",
                 "veg": veg, "outcome": mo, "food_argv": food_argv}
        np.savez_compressed(mdir / f"ep_{seed:05d}{tag}.npz", images_front=np.stack([r_["front"] for r_ in Rm]),
                            images_wrist=np.stack([r_["wrist"] for r_ in Rm]), state=np.stack([r_["state"] for r_ in Rm]),
                            cmd=np.stack([r_["cmd"] for r_ in Rm]), t=np.array([r_["t"] for r_ in Rm]),
                            block=np.array([*pp, 0.0, *(np.ptp(loc, axis=0))], np.float32), cmd_steps=np.stack(run.cmd_steps),
                            step_phase=np.array([r_["phase"] for r_ in run.rows]), tick_phase=np.array([r_.get("phase", "") for r_ in Rm]), grip_cmd=np.array([r_.get("grip", np.nan) for r_ in Rm], np.float32))
        (mdir / f"ep_{seed:05d}{tag}.json").write_text(json.dumps(minfo, indent=1, default=float))
        print(f"[food_robot] move seed {seed}: " + json.dumps({k_: v for k_, v in mo.items() if k_ != "candidates"}, default=float), flush=True)
        if e < 2 or a.task_kind == "move_replay":
            import imageio.v2 as imageio
            for n_ in ("front", "wrist"):
                for i_ in np.linspace(0, len(Rm) - 1, 6).astype(int):
                    (mdir / "frames").mkdir(exist_ok=True)
                    imageio.imwrite(mdir / "frames" / f"ep{seed:05d}_{n_}_{i_:03d}.png", Rm[i_][n_])
        run.rec, run.cmd_steps, run.rows = None, None, []
        return mo

    def piece_points(objs, recs, idx):
        """조각 idx 들의 지금 월드 점(입자 자리를 강체 자세로 옮김)."""
        from scipy.spatial.transform import Rotation
        out_ = []
        for i_ in idx:
            pp_, pq_ = (T.np_(v) for v in objs[i_].get_position_orientation())
            out_.append(pp_ + recs[i_]["local"].astype(float) @ Rotation.from_quat(pq_).as_matrix().T)
        return out_

    video_marks = []  # 11-26 reports/79 --video_all: (이름, 장면 시간 s, 프레임 폴더 이름, 그 폴더 안 프레임 번호)

    def vmark(name):
        if a.video_all and a.overview:
            video_marks.append((name, round(run.k * dt, 3), run.overview_dir.name if run.overview_dir is not None else None, int(run.overview_n)))

    def aim_overview(target, eye_off, sub=None):
        """영상용 넓은 카메라를 target 을 보게(eye = target + eye_off), sub 이면 그 폴더로 새로 찍기 시작."""
        if "overview" not in cams:
            return
        eye_ = np.asarray(target, float) + np.asarray(eye_off, float)
        cams["overview"].set_position_orientation(position=th.tensor(eye_, dtype=th.float32),
                                                  orientation=th.tensor(np.asarray(T.look_at(eye_, np.asarray(target, float)), float), dtype=th.float32))
        if sub is not None:
            run.overview_every = max(1, round(hz / a.overview_fps))
            run.overview_dir = out / "overview_frames" / sub
            run.overview_dir.mkdir(parents=True, exist_ok=True)
            run.overview_n = 0
        for _ in range(10):
            og.sim.render()

    def in_plate(P_):
        """조각 점 P_ 가 접시(둥근 접시 반지름 75 mm / 네모 접시 안쪽) 위·안에 있나."""
        c_ = P_.mean(axis=0)
        if use_bowl:
            return bool(np.linalg.norm(c_[:2] - np.array(PLATE_XY)) < a.bowl[0] - 0.008 and TABLE_H < P_[:, 2].min() < TABLE_H + a.bowl[1] + 0.01)
        if use_tray:
            ins = abs(c_[0] - PLATE_XY[0]) < 0.5 * a.tray[0] - 0.004 and abs(c_[1] - PLATE_XY[1]) < 0.5 * a.tray[1] - 0.004
            return bool(ins and TABLE_H < P_[:, 2].min() < TABLE_H + a.tray[2])
        return bool(np.linalg.norm(c_[:2] - np.array(PLATE_XY)) < 0.075 and TABLE_H + 0.0 < P_[:, 2].min() < TABLE_H + 0.045)

    def rot_axis(axis, ang):
        from scipy.spatial.transform import Rotation
        return Rotation.from_rotvec(np.asarray(axis, float) * float(ang)).as_matrix()

    def do_unstick(objs, recs, glued, seed, e, veg, K_in_E_, mon=None):
        """과제 ④: 칼에 붙은 편 떼기. 칼을 쥔 채 썰기 끝 자리에서 시작(--unstick 으로 방법 고름).
        shake(11-25 reports/77): 들기 → 붙은 편을 접시 위로 → 위아래 털기. tap/scrape/wipe(11-25 밤 reports/78): 도마 내리치기 / 네모 접시 벽에 긁기 /
        도움 팔 손끝으로 쓸어내리기. 접착은 --glue_model(기본 slide: 미끄러지는 D6 이음 — 손가락·벽·충격 모두 PhysX 가 같은 식으로 풂).
        → 결과 dict(붙인 편 없으면 None)."""
        import food_handoff as FH
        if not glued:
            return None
        method = a.unstick
        udir = out / "unstick"
        udir.mkdir(exist_ok=True)
        E_in_K_ = np.linalg.inv(K_in_E_)
        for _ in range(15):  # DLSS 가 앞 장면을 섞어 그리지 않게
            og.sim.render()
        run.rec, run.cmd_steps, run.rows = [], [], []
        gk = [gl["k"] for gl in glued]
        slide = hasattr(mon, "attached")
        hl, kh = float(knife_geo.half_length), float(knife_geo.height)

        def att_names():
            if slide:
                return set(mon.attached())
            st_ = FH.glue_state(knife, objs, glued)
            return {s_["name"] for s_ in st_ if s_["attached"] and (mon is None or s_["name"] not in mon.broken)}

        def n_attached():
            return len(att_names())

        def att_k():
            nm_ = att_names()
            return [gl["k"] for gl in glued if gl["name"] in nm_]

        def hold(sec, K_, phase):
            run.phase = phase
            for _ in range(int(round(sec / dt))):
                run.phase = phase
                run.step(K_ @ E_in_K_, T.CLOSE, K_cmd=K_)

        def glued_in_K(K_):
            """붙어 있는 편 점들(칼 틀 K_)."""
            Ki_ = np.linalg.inv(K_)
            ks_ = att_k()
            return [p_ @ Ki_[:3, :3].T + Ki_[:3, 3] for p_ in piece_points(objs, recs, ks_)] if ks_ else []

        sides = [gl["side"] for gl in glued]
        s_ = 1 if sides.count(1) >= sides.count(-1) else -1  # 붙은 편이 많은 면

        def closeup(target, eye_off):
            """영상용 넓은 카메라(--overview)를 이 동작 자리로 돌려 과제 동안 프레임 저장(학습 입력 아님)."""
            if "overview" not in cams:
                return
            if getattr(run, "_ov_pose0", None) is None:  # 과제 끝에 되돌릴 원래 자세(실제 파이프라인 다음 에피소드 썰기 영상용)
                run._ov_pose0 = cams["overview"].get_position_orientation()
            eye_ = np.asarray(target, float) + np.asarray(eye_off, float)
            cams["overview"].set_position_orientation(position=th.tensor(eye_, dtype=th.float32),
                                                      orientation=th.tensor(np.asarray(T.look_at(eye_, np.asarray(target, float)), float), dtype=th.float32))
            run.overview_every = max(1, round(hz / a.overview_fps))
            run.overview_dir = out / "overview_frames" / f"unstick_{method}_ep{seed:05d}"
            run.overview_dir.mkdir(parents=True, exist_ok=True)
            run.overview_n = 0
            for _ in range(10):
                og.sim.render()
        kz, hist, extra = [], [], {"side": s_, "sides": sides}
        if slide:  # 붙을 때 칼 면에 닿은 칸 범위(칼 틀 x: −뒤꿈치 … +칼끝, z: 날 끝 0 … 칼등)
            extra["glued_x_mm"] = [[round(1e3 * float(gl["cells_xz"][:, 0].min()), 1), round(1e3 * float(gl["cells_xz"][:, 0].max()), 1)] for gl in glued]
            extra["glued_z_mm"] = [[round(1e3 * float(gl["cells_xz"][:, 1].min()), 1), round(1e3 * float(gl["cells_xz"][:, 1].max()), 1)] for gl in glued]
        K0 = run.knife_pose()
        ns0 = s_ * K0[:3, 1].copy()
        ns0[2] = 0.0
        ns0 /= np.linalg.norm(ns0)
        if method == "tap":  # 도마 위 칠 자리 옆(붙은 면 쪽·칼끝 쪽)에서
            closeup(K0[:3, 3] + np.array([0, 0, -0.02]) + 0.06 * ns0, 0.30 * ns0 + np.array([0.0, -0.16, 0.14]))
        elif method == "scrape":  # 그릇 안쪽(붙은 면 반대쪽)·칼끝 쪽 위에서 벽을 내려다봄
            closeup(np.array([PLATE_XY[0] + 0.5 * a.tray[0] * np.sign(ns0[0]), PLATE_XY[1], TABLE_H + 0.06]),
                    -0.30 * ns0 + np.array([0.0, -0.20, 0.26]))
        elif method == "wipe":  # 붙은 면 쪽(+ns) 옆에서 — 도움 팔은 앞(−y), 첫 팔은 뒤(+y)에서 옴
            closeup(np.array([PLATE_XY[0], PLATE_XY[1], PLATE_TOP + 0.09]), 0.32 * ns0 + np.array([0.0, -0.08, 0.08]))
        else:  # shake: 접시 위 칼 면(붙은 면 쪽)을 앞쪽에서
            closeup(np.array([PLATE_XY[0], PLATE_XY[1], PLATE_TOP + 0.09]), 0.22 * ns0 + np.array([0.0, -0.30, 0.10]))
        K1 = K0.copy()
        K1[2, 3] += 0.08
        T.move_knife(run, K_in_E_, K1, T.CLOSE, speed=0.06, settle=0.1, phase="lift")
        hist.append(n_attached())

        def aligned(K_):
            """칼 길이(x)를 월드 ±y(지금 x 의 y 부호)로, 면 법선(y)을 월드 ±x 로 맞춘 회전(접시 벽·도움 팔과 나란히)."""
            xs_ = np.array([0.0, -1.0 if K_[1, 0] < 0 else 1.0, 0.0])
            zs_ = np.array([0.0, 0.0, 1.0])
            return np.stack([xs_, np.cross(zs_, xs_), zs_], axis=1)

        if method == "shake":
            amp, shz, cyc, rounds = float(a.shake[0]), float(a.shake[1]), int(a.shake[2]), int(a.shake[3])  # shz: 털기 빈도(물리 hz 와 다른 이름)
            K2 = run.knife_pose()
            ks_ = att_k()
            if ks_:
                Pa = np.concatenate(piece_points(objs, recs, ks_))
                cen, zlo = Pa[:, :2].mean(axis=0), float(Pa[:, 2].min())
            else:
                cen, zlo = K2[:2, 3], float(K2[2, 3])
            K2[:2, 3] += np.array(PLATE_XY) - cen
            K2[2, 3] += (PLATE_TOP + 0.06) - zlo
            T.move_knife(run, K_in_E_, K2, T.CLOSE, speed=0.08, settle=0.3, phase="carry")
            hist[0] = n_attached()
            n_rounds = 0
            for rnd in range(rounds):
                if hist[-1] == 0:
                    break
                n_rounds += 1
                n = int(round(cyc / shz / dt))
                amp_r = amp * (1.0 + 0.2 * rnd)  # 안 떨어지면 판마다 20 % 세게
                for i in range(1, n + 1):
                    Kt = K2.copy()
                    Kt[2, 3] -= amp_r * 0.5 * (1.0 - math.cos(2.0 * math.pi * shz * i * dt))
                    run.phase = "shake"
                    run.step(Kt @ E_in_K_, T.CLOSE, K_cmd=Kt)
                    kz.append(float(run.knife_pose()[2, 3]))
                hold(0.3, K2, "check")
                hist.append(n_attached())
            extra["rounds"] = n_rounds
            hold(0.8, K2, "settle")
            K3 = K2.copy()
            K3[2, 3] += 0.04
            T.move_knife(run, K_in_E_, K3, T.CLOSE, speed=0.08, settle=0.2, phase="retreat")
        elif method == "tap":
            spd, ntap, pitch, h0 = float(a.tap[0]), int(a.tap[1]), math.radians(float(a.tap[2])), float(a.tap[3])
            Kc = run.knife_pose()
            ns = s_ * Kc[:3, 1].copy()
            ns[2] = 0.0
            ns /= np.linalg.norm(ns)
            # 칠 자리: 붙은 편이 떨어질 쪽(ns)으로, 도마 위 다른 조각(몸통·썬 편)보다 25 mm 더 나간 곳(칼날 길이 안의 점만 셈)
            O = Kc[:3, 3]
            far = 0.0
            for i_, P_ in enumerate(piece_points(objs, recs, range(len(objs)))):
                if i_ in gk:
                    continue
                d_ = (P_ - O)[:, :2]
                along = np.abs(d_ @ Kc[:2, 0] / max(1e-9, np.linalg.norm(Kc[:2, 0])))
                sel = along < hl + 0.02
                if sel.any():
                    far = max(far, float((d_[sel] @ ns[:2]).max()))
            dsh = max(0.0, far) + 0.025
            # 숙임 각: 붙은 편의 칼끝 쪽 끝에서 날 끝이 도마 위 15 mm 이상 뜨게(편이 칼끝 가까이 붙으면 더 숙임, 12~35°) — 11-25 밤 t2: 12° 로는
            #   칼끝에서 53 mm 인 편 밑 날 끝이 11 mm 만 떠, 미끄러진 편 귀퉁이가 도마에 막힘(첫 번째만 떨어짐)
            gK0 = glued_in_K(Kc)
            x_tipend = max([float(g_[:, 0].max()) for g_ in gK0] + [-hl])
            need = math.asin(min(0.95, 0.015 / max(0.02, hl - x_tipend)))
            pitch = float(np.clip(max(pitch, need), pitch, math.radians(35.0)))
            extra.update(pitch_deg=round(math.degrees(pitch), 1), pieces_tip_end_from_tip_mm=round(1e3 * (hl - x_tipend), 1))
            Kt = Kc.copy()
            Kt[:2, 3] = O[:2] + dsh * ns[:2]
            Kt[:3, :3] = Kc[:3, :3] @ rot_axis([0.0, 1.0, 0.0], pitch)  # 칼끝(+x)을 pitch 만큼 숙임(칼 면 법선 축으로)
            tip_w = Kt[:3, :3] @ np.array([hl, 0.0, 0.0]) + Kt[:3, 3]
            Kt[2, 3] += (ZB + h0) - tip_w[2]
            land = Kt[:2, 3] + 0.02 * ns[:2]
            extra.update(tap_shift_mm=round(1e3 * dsh, 1), tap_xy=np.round(Kt[:2, 3], 4).tolist(),
                         land_on_board=bool(abs(land[0]) < 0.17 - 0.01 and abs(land[1]) < 0.12 - 0.01),
                         edge_above_board_at_pieces_mm=None)
            Kup = Kt.copy()
            Kup[2, 3] += 0.03
            T.move_knife(run, K_in_E_, Kup, T.CLOSE, speed=0.08, ang_speed=0.5, settle=0.2, phase="to_tap")
            T.move_knife(run, K_in_E_, Kt, T.CLOSE, speed=0.06, settle=0.3, phase="aim")
            gK = glued_in_K(run.knife_pose())
            if gK:  # 편 자리(칼 길이 x 가운데)에서 날 끝이 도마 위 얼마나 떴나(칼끝이 도마에 닿을 때)
                xm = float(np.mean([g_[:, 0].mean() for g_ in gK]))
                extra["edge_above_board_at_pieces_mm"] = round(1e3 * (hl - xm) * math.sin(pitch), 1)
            hits = []
            for r in range(ntap):
                if hist[-1] == 0:
                    break
                drop, z_off, t_ = h0 + 0.005, 0.0, 0.0
                while z_off < drop:
                    t_ += dt
                    z_off = min(drop, z_off + spd * min(1.0, t_ / 0.08) * dt)
                    Kd = Kt.copy()
                    Kd[2, 3] -= z_off
                    run.phase = "tap_down"
                    run.step(Kd @ E_in_K_, T.CLOSE, K_cmd=Kd)
                    kz.append(float(run.knife_pose()[2, 3]))
                tip_now = run.knife_pose()[:3, :3] @ np.array([hl, 0.0, 0.0]) + run.knife_pose()[:3, 3]
                hits.append(round(1e3 * float(tip_now[2] - ZB), 2))
                vz_ = np.diff(np.array(kz[-12:])) / dt if len(kz) > 12 else np.zeros(1)
                extra.setdefault("tap_vz_min_mps", []).append(round(float(vz_.min()), 3))  # 부딪기 직전 칼 아래로 빠르기(가장 빠른 단계)
                hold(0.06, Kd, "tap_hit")
                T.move_knife(run, K_in_E_, Kt, T.CLOSE, speed=0.15, settle=0.15, phase="tap_up")
                hist.append(n_attached())
            extra.update(taps=len(hist) - 1, tip_above_board_at_hit_mm=hits)
            hold(0.3, Kt, "check")
            K3 = Kt.copy()
            K3[2, 3] += 0.04
            T.move_knife(run, K_in_E_, K3, T.CLOSE, speed=0.08, settle=0.2, phase="retreat")
        elif method == "scrape":
            tilt, spd, gap = math.radians(float(a.scrape[0])), float(a.scrape[1]), float(a.scrape[2])
            Kc = run.knife_pose()
            R_al = aligned(Kc)
            R_t = R_al @ rot_axis([1.0, 0.0, 0.0], -s_ * tilt)  # 붙은 면(s_·y)이 아래를 보게
            ns = s_ * R_al[:, 1]  # 붙은 면 바깥 법선(기울이기 전, 수평 = 월드 ±x)
            ez_t = R_t[:, 2]
            gK = glued_in_K(Kc)
            ztop = max([float(g_[:, 2].max()) for g_ in gK] + [kh])
            z0 = ztop + 0.008  # 벽 모서리가 처음에 칼등·가장 높은 편보다 8 mm 위(칼 틀) — 튀어나온 편(3~4 mm)이 벽 안쪽 면과 안 겹치게
            xm = float(np.mean([g_[:, 0].mean() for g_ in gK])) if gK else 0.0
            wall_x = PLATE_XY[0] + np.sign(ns[0]) * (0.5 * a.tray[0] - 0.006)  # 벽 안쪽 윗모서리(붙은 면 쪽 벽)
            zc = TABLE_H + a.tray[2]
            ht0 = float(FH.blade_half_thickness(min(z0, kh), knife_geo.edge_thickness, knife_geo.spine_thickness, knife_geo.bevel_height))
            Pc = np.array([wall_x, 0.0, zc])
            Ks = np.eye(4)
            Ks[:3, :3] = R_t
            Ks[:3, 3] = Pc - R_t @ np.array([0.0, s_ * (ht0 + gap), z0])
            # 칼 자리(월드 y): 칼 뒤꿈치 3 mm 안(칼 틀 x = −hl + 3 mm)을 높은 벽의 +y 끝에 맞춤 → 손잡이·로봇 손은 낮은 턱 위, 편은 높은 벽에 긁힘
            #   (칼날 8 cm, 편이 뒤꿈치 쪽 −42~+15 mm 에 붙음 — 11-25 밤 s4~s7). 칼 길이 축이 −y 를 볼 때(뒤꿈치 +y)만 맞음.
            xs_y = float(R_t[1, 0])  # 칼 길이 축의 월드 y(±1)
            if xs_y > 0:
                print("[food_robot] [경고] scrape: 칼 뒤꿈치가 −y 쪽 — 접시 낮은 턱(+y 끝)과 반대라 손잡이가 벽에 걸릴 수 있음", flush=True)
            wall_end_y = PLATE_XY[1] + 0.5 * a.tray[1] - a.tray_gap
            heel_need = wall_end_y
            oy = wall_end_y + xs_y * (hl - 0.003)
            Ks[1, 3] = oy
            extra["pieces_y_from_tray_centre_mm"] = round(1e3 * (oy + xs_y * xm - PLATE_XY[1]), 1)
            extra["scrape_dbg"] = {"xm_mm": round(1e3 * xm, 1), "xs_y": xs_y, "oy": round(oy, 4), "heel_need": round(heel_need, 4), "hl": hl,
                                   "R_t": np.round(R_t, 3).tolist()}
            L = z0 + 0.008  # 모서리가 날 끝보다 8 mm 지나도록
            # 처음 자세에서 날 끝·편 가장 낮은 점이 접시 바닥보다 얼마나 위인가(바닥에 닿으면 안 됨)
            edge_lo = min(float((R_t @ np.array([x_, 0.0, 0.0]) + Ks[:3, 3])[2]) for x_ in (-hl, hl))
            pieces_lo = min([float((g_ @ R_t.T + Ks[:3, 3])[:, 2].min()) for g_ in gK] + [edge_lo])
            extra.update(tilt_deg=round(math.degrees(tilt), 1), wall_x=round(float(wall_x), 4), z_start_on_blade_mm=round(1e3 * z0, 1),
                         edge_above_floor_mm=round(1e3 * (edge_lo - PLATE_TOP), 1), pieces_above_floor_mm=round(1e3 * (pieces_lo - PLATE_TOP), 1))
            Kpre = Ks.copy()
            Kpre[:3, 3] += 0.05 * np.array([0.0, 0.0, 1.0]) - 0.03 * ns
            T.move_knife(run, K_in_E_, Kpre, T.CLOSE, speed=0.08, ang_speed=0.5, settle=0.3, phase="carry")
            T.move_knife(run, K_in_E_, Ks, T.CLOSE, speed=0.04, settle=0.3, phase="aim")
            hist.append(n_attached())
            Ke = Ks.copy()
            Ke[:3, 3] += L * ez_t
            T.move_knife(run, K_in_E_, Ke, T.CLOSE, speed=spd, settle=0.4, phase="scrape")
            hist.append(n_attached())
            K3 = Ke.copy()
            K3[2, 3] += 0.05
            T.move_knife(run, K_in_E_, K3, T.CLOSE, speed=0.08, settle=0.3, phase="retreat")
        elif method == "wipe":
            spd, gap, stp = float(a.wipe[0]), float(a.wipe[1]), float(a.wipe[2])
            if run.helper_arm is None:
                raise RuntimeError("--unstick wipe 인데 도움 팔이 없음(--task_kind move/unstick_replay 에서만 만듦)")
            ha = run.helper_arm
            Kc = run.knife_pose()
            Kw = np.eye(4)
            Kw[:3, :3] = aligned(Kc)
            ns = s_ * Kw[:3, 1]
            if ns[0] < 0:
                print("[food_robot] [경고] wipe: 붙은 편 면이 도움 팔 반대쪽(−x)을 봄 — 도움 팔이 칼 너머로 손을 뻗어야 함", flush=True)
            gK = glued_in_K(Kc)
            Pa = np.concatenate([g_ for g_ in gK]) if gK else np.zeros((1, 3))
            xlo, xhi = float(Pa[:, 0].min()), float(Pa[:, 0].max())
            zlo, zhi = float(Pa[:, 2].min()), float(Pa[:, 2].max())
            xm = 0.5 * (xlo + xhi)
            # 칼: 붙은 편 가운데가 접시 가운데 위, 편 가장 낮은 점이 접시 위 60 mm
            Kw[:3, 3] = np.array([PLATE_XY[0], PLATE_XY[1], 0.0]) - Kw[:3, :3] @ np.array([xm, 0.0, 0.0])
            Kw[2, 3] = PLATE_TOP + 0.06 - zlo
            run.helper_cmd = (run.helper_cmd[0], T.OPEN)  # 칼을 나르는 동안 도움 팔 집게를 다 벌려 둠(w10: 덜 벌어진 채 재서 손가락이 칼 면에서 8 mm 떨어짐)
            T.move_knife(run, K_in_E_, Kw, T.CLOSE, speed=0.08, ang_speed=0.5, settle=0.6, phase="carry")
            hist[0] = n_attached()
            ffh = finger_frame(T, ha.robot, ha)  # 도움 팔 손끝 틀(잡는 점 = 손끝보다 9 mm 위)
            tip_off = 0.009  # 잡는 점 → 손가락 끝
            fhalf = 0.009  # 닫은 두 손가락 반두께(칼 면 법선 쪽)
            # 손가락은 칼 면 법선(ns) 쪽으로 닫힘 → 손(닫히는 쪽 약 20 cm)이 칼날을 가로질러 칼등 위에 놓이고, 칼 길이 쪽으로는 ±3 cm 만 차지
            #   (칼 길이 쪽으로 닫으면 손이 칼 뒤꿈치 너머 첫 로봇 손과 부딪혀 칼이 67 mm 밀림 — 11-25 밤 w3)
            close_d = ns.copy()
            x_min_ok = -hl + 0.025  # 손잡이 쪽 첫 로봇 손(칼 틀 x ≈ −hl − 0.055 ± 0.032)과 떨어지게(w8: −hl + 0.02 에서 두 손 사이 11 mm → 손 들 때 칼이 74 mm 밀림)
            x_a, x_b = max(xlo + 0.009, x_min_ok), max(min(xhi - 0.009, hl - 0.01), max(xlo + 0.009, x_min_ok))
            n_str = max(1, int(math.ceil((x_b - x_a) / stp)) + 1) if x_b - x_a > 1e-3 else 1
            xs_list = [0.5 * (x_a + x_b)] if n_str == 1 else list(np.linspace(x_a, x_b, n_str))
            xs_list = xs_list[::-1]  # 칼끝 쪽부터(손잡이 쪽으로 다가가며)
            z_start, z_end = zhi + 0.008, 0.004
            strokes = []

            geo = {"near": -fhalf, "tip": tip_off, "hand": 0.054, "measured": False}  # 잡는 점 기준: 칼 쪽 손가락 면(ns 성분)·손끝 아래·손 바닥 위

            def measure_fingers():
                """도움 팔 손가락·손의 실제 충돌 점으로 잡는 점 기준 거리를 잰다(11-25 밤 w4: 두 손가락 반두께를 9 mm 로 어림 → 칼 면·칼등을 눌러 칼 24 mm 밀림)."""
                E_h = ha.eef()
                cw = E_h[:3, :3] @ ffh[0] + E_h[:3, 3]
                fp = [T.np_(lk.collision_boundary_points_world) for lk in ha.robot.finger_links[ha.arm] if lk.collision_boundary_points_world is not None]
                hp = [T.np_(lk.collision_boundary_points_world) for nm_, lk in ha.robot.links.items()
                      if "hand" in nm_ and lk.collision_boundary_points_world is not None]
                if fp:
                    P_ = np.concatenate(fp)
                    zlo_ = float(P_[:, 2].min())
                    # 편 쪽(+ns) 손가락 = 점들의 ns 평균이 큰 쪽, 그 손가락 안쪽 면(−ns 쪽 = 칼 쪽) = 그 점들 ns 최솟값(손끝 30 mm 안)
                    fing = max(fp, key=lambda q_: float(((q_ - cw) @ ns).mean()))
                    fing = fing[fing[:, 2] < zlo_ + 0.03]
                    geo["near"] = float(((fing - cw) @ ns).min())
                    other = min(fp, key=lambda q_: float(((q_ - cw) @ ns).mean()))
                    geo["far_inner"] = float(((other - cw) @ ns).max())  # 반대 손가락 안쪽 면(칼 반대 면에서 떨어져 있어야)
                    geo["tip"] = float(cw[2] - zlo_)
                if hp:
                    geo["hand"] = float(np.concatenate(hp)[:, 2].min() - cw[2])
                geo["measured"] = True
                extra["helper_finger_geom_mm"] = {k_: round(1e3 * v_, 1) for k_, v_ in geo.items() if k_ != "measured"}

            def fpt(x_, z_):
                """손가락 가장 낮은 점이 칼 틀 높이 z_, 칼 쪽 손가락 면이 칼 면에서 gap 밖(칼 틀 x_)이 되는 '잡는 점' 월드 자리.
                칼 면 자리는 손가락이 걸치는 높이(z_ … z_ + 20 mm) 중 가장 두꺼운 곳으로."""
                ht_ = float(FH.blade_half_thickness(max(0.0, min(z_ + 0.02, kh)), knife_geo.edge_thickness, knife_geo.spine_thickness, knife_geo.bevel_height))
                face_ = Kw[:3, :3] @ np.array([x_, s_ * ht_, z_]) + Kw[:3, 3]
                return face_ + (gap - geo["near"]) * ns + np.array([0.0, 0.0, geo["tip"]])

            def helper_to(E1, speed, phase, settle=0.0):
                E0 = run.helper_cmd[0]
                dist = float(np.linalg.norm(E1[:3, 3] - E0[:3, 3]))
                ang = math.radians(T.rotation_deg(E0[:3, :3], E1[:3, :3]))
                n = int(math.ceil(max(0.3, dist / speed, ang / 0.8) / dt))
                for i in range(1, n + 1):
                    run.helper_cmd = (T.interpolate(E0, E1, T.min_jerk(i / n)), hgrip)
                    run.phase = phase
                    run.step(Kw @ E_in_K_, T.CLOSE, K_cmd=Kw)
                for _ in range(int(round(settle / dt))):
                    run.phase = phase
                    run.step(Kw @ E_in_K_, T.CLOSE, K_cmd=Kw)

            # 11-25 밤 w7: 손가락 뒷면(뿌리가 12 mm 더 두꺼움)으로 쓸면 뿌리가 칼 면을 스치고(w6), 기울이면 손 끝이 칼등으로 내려감(w7)
            #   → 집게를 벌려 칼날을 두 손가락 사이에 두고, 편 쪽(+ns) 손가락의 평평한 안쪽 면으로 쓸어내림(엄지·검지로 칼날을 훑는 것과 같음)
            def eef_at(p_):
                return topdown_eef(p_, close_d, ffh)

            E_home = run.helper_cmd[0].copy()
            hgrip = T.OPEN
            for rnd in range(2):
                if n_attached() == 0:
                    break
                for x_ in xs_list:
                    if n_attached() == 0:
                        break
                    p_hi = fpt(x_, z_start) + np.array([0.0, 0.0, 0.04])
                    helper_to(eef_at(p_hi), 0.15, "helper_reach", settle=0.1)
                    if not geo["measured"]:  # 손가락 실제 크기를 재고, 손 바닥이 칼등 위 5 mm 밑으로 안 내려가게 쓸기 끝을 정함
                        measure_fingers()
                        z_end = max(z_end, kh + 0.005 - (geo["hand"] + geo["tip"]))
                        extra["wipe_z_end_mm"] = round(1e3 * z_end, 1)
                        p_hi = fpt(x_, z_start) + np.array([0.0, 0.0, 0.04])
                        helper_to(eef_at(p_hi), 0.10, "helper_reach", settle=0.1)
                    helper_to(eef_at(fpt(x_, z_start)), 0.06, "helper_down", settle=0.1)
                    n0_ = n_attached()
                    helper_to(eef_at(fpt(x_, z_end)), spd, "wipe", settle=0.2)
                    eh_ = ha.eef()
                    strokes.append({"x_on_blade_mm": round(1e3 * x_, 1), "attached_before": n0_, "attached_after": n_attached(),
                                    "finger_err_mm": round(1e3 * float(np.linalg.norm(eh_[:3, 3] - run.helper_cmd[0][:3, 3])), 2)})
                    helper_to(eef_at(fpt(x_, z_end) + 0.02 * ns), 0.06, "helper_away", settle=0.0)
                    helper_to(eef_at(fpt(x_, z_start) + 0.02 * ns + np.array([0.0, 0.0, 0.04])), 0.12, "helper_up", settle=0.0)
                hist.append(n_attached())
            E_up = run.helper_cmd[0].copy()
            E_up[2, 3] += 0.10
            helper_to(E_up, 0.15, "helper_up", settle=0.0)  # 곧장 위로 10 cm 뒤 집으로(첫 로봇 팔 옆을 낮게 지나지 않게)
            hgrip = T.CLOSE
            helper_to(E_home, 0.2, "helper_home", settle=0.3)
            extra.update(strokes=strokes, n_strokes_per_round=len(xs_list))
            hold(0.5, Kw, "settle")
            K3 = Kw.copy()
            K3[2, 3] += 0.04
            T.move_knife(run, K_in_E_, K3, T.CLOSE, speed=0.08, settle=0.2, phase="retreat")
        else:
            raise ValueError(method)

        run.overview_every = 0
        if getattr(run, "_ov_pose0", None) is not None:  # 영상용 카메라를 원래 자리로
            cams["overview"].set_position_orientation(position=run._ov_pose0[0], orientation=run._ov_pose0[1])
            run._ov_pose0 = None
        Pg = piece_points(objs, recs, gk)
        still = att_names()
        extra["final_centre_xyz"] = [np.round(P_.mean(axis=0), 4).tolist() for P_ in Pg]
        where = []
        for gl, P_ in zip(glued, Pg):
            c_ = P_.mean(axis=0)
            on_board = bool(abs(c_[0]) < 0.17 and abs(c_[1]) < 0.12 and P_[:, 2].min() < ZB + 0.03)
            where.append("knife" if gl["name"] in still else "plate" if in_plate(P_) else "board" if on_board else "table")
        acc = np.diff(np.array(kz), 2) / dt / dt if len(kz) > 2 else np.zeros(1)
        goal = "board" if method == "tap" else "plate"
        uo = {"method": method, "glue_model": a.glue_model, "glued": [gl["name"] for gl in glued], "break_N": [round(gl["break_N"], 4) for gl in glued],
              "attached_after_lift_carry": hist[0], "attached_hist": hist, "detached": int(len(glued) - len(still)), "where": where,
              "on_plate": int(sum(w_ == "plate" for w_ in where)), "on_goal": int(sum(w_ == goal for w_ in where)), "goal": goal,
              "knife_acc_max_mps2": round(float(np.abs(acc).max()), 2), "ticks": len(run.rec),
              "monitor": mon.summary() if mon is not None else None, **extra}
        uo["success"] = bool(uo["detached"] == len(glued) and uo["on_goal"] == len(glued))
        if mon is not None and hasattr(mon, "trace"):  # 미끄럼 자취(4 단계마다: 단계, sx mm, sz mm, sy mm, 넓이 비율, 늘어남 힘 N) — 단계 번호는 감시기 기준
            uo["slide_trace"] = mon.trace
            uo["phase_at_step"] = getattr(mon, "phase_log", None)
        R_ = run.rec
        task_ = {"shake": f"shake the {veg.replace('_', ' ')} slices off the knife onto the plate",
                 "tap": f"tap the knife on the board to knock the {veg.replace('_', ' ')} slices off",
                 "scrape": f"scrape the {veg.replace('_', ' ')} slices off the knife on the edge of the dish",
                 "wipe": f"wipe the {veg.replace('_', ' ')} slices off the knife with the other hand"}[method]
        info_ = {"seed": seed, "look": LOOK.current, "mode": f"expert_unstick_{method}", "task": a.task or task_, "veg": veg, "outcome": uo, "food_argv": food_argv}
        sv = dict(images_front=np.stack([r_["front"] for r_ in R_]), images_wrist=np.stack([r_["wrist"] for r_ in R_]),
                  state=np.stack([r_["state"] for r_ in R_]), cmd=np.stack([r_["cmd"] for r_ in R_]), t=np.array([r_["t"] for r_ in R_]),
                  block=np.zeros(7, np.float32), cmd_steps=np.stack(run.cmd_steps),
                  step_phase=np.array([r_["phase"] for r_ in run.rows]), tick_phase=np.array([r_.get("phase", "") for r_ in R_]), grip_cmd=np.array([r_.get("grip", np.nan) for r_ in R_], np.float32))
        if R_ and "helper_state" in R_[0]:
            sv.update(helper_state=np.stack([r_["helper_state"] for r_ in R_]), helper_cmd=np.stack([r_["helper_cmd"] for r_ in R_]))
        np.savez_compressed(udir / f"ep_{seed:05d}.npz", **sv)
        (udir / f"ep_{seed:05d}.json").write_text(json.dumps(info_, indent=1, default=float))
        print(f"[food_robot] unstick seed {seed}: " + json.dumps(uo, default=float), flush=True)
        import imageio.v2 as imageio
        for n_ in ("front", "wrist"):
            for i_ in np.linspace(0, len(R_) - 1, 8).astype(int):
                (udir / "frames").mkdir(exist_ok=True)
                imageio.imwrite(udir / "frames" / f"ep{seed:05d}_{n_}_{i_:03d}.png", R_[i_][n_])
        run.rec, run.cmd_steps, run.rows = None, None, []
        return uo

    def do_separate(objs, recs, seed, e, veg, K_in_E_, K_cut):
        """과제 ⑤(11-25 reports/77): 몸통에 붙어 줄지어 선 편을 칼로 밀어 떼어 놓기(눕히기). 칼을 쥔 채.
        K_cut = 강체로 바꿀 때 칼 자세(마지막 칼질 자리, 칼 면 +y 쪽이 잘려 나간 편). 대상 = 몸통 아닌, 높이 ≥ 12 mm, 도마에 놓인(바닥 ≤ 도마 + 3 mm),
        칼 면에서 +y 쪽 8 mm 안에 점이 있는 조각. 칼을 그 자리 위로 → 칼 틈으로 날 끝을 도마 위 4 mm 까지 내림 → +y 로 18 mm 밀기 → 들기.
        → 결과 dict(대상 없으면 None)."""
        from scipy.spatial.transform import Rotation
        body_ = max(range(len(recs)), key=lambda i_: recs[i_]["n"])
        Ki = np.linalg.inv(K_cut)
        Pw = piece_points(objs, recs, range(len(objs)))
        tgt = []
        for i_, P_ in enumerate(Pw):
            if i_ == body_:
                continue
            h_ = float(P_[:, 2].max() - P_[:, 2].min())
            L_ = P_ @ Ki[:3, :3].T + Ki[:3, 3]
            if h_ >= 0.012 and P_[:, 2].min() <= ZB + 0.003 and np.any((L_[:, 1] > 0) & (L_[:, 1] < 0.008)) and np.abs(L_[:, 0]).min() < 0.045:
                tgt.append(i_)
        if not tgt:
            return None
        sdir = np.array(K_cut[:3, 1], float)  # 칼 면 +y(월드) = 잘려 나간 편 쪽
        sdir[2] = 0.0
        sdir /= np.linalg.norm(sdir)
        top = max(float(P_[:, 2].max()) for P_ in Pw)
        for _ in range(15):
            og.sim.render()
        run.rec, run.cmd_steps, run.rows = [], [], []
        Ka = K_cut.copy()
        Ka[2, 3] = top + 0.02  # 날 끝이 가장 높은 조각보다 20 mm 위
        T.move_knife(run, K_in_E_, Ka, T.CLOSE, speed=0.12, settle=0.2, phase="approach")
        Kb = K_cut.copy()
        Kb[2, 3] = ZB + 0.004
        T.move_knife(run, K_in_E_, Kb, T.CLOSE, speed=0.04, settle=0.2, phase="descend")
        P0 = [p_.copy() for p_ in piece_points(objs, recs, range(len(objs)))]
        Kc = Kb.copy()
        Kc[:3, 3] += 0.018 * sdir
        T.move_knife(run, K_in_E_, Kc, T.CLOSE, speed=0.03, settle=0.3, phase="push")
        Kd = Kc.copy()
        Kd[2, 3] = top + 0.03
        T.move_knife(run, K_in_E_, Kd, T.CLOSE, speed=0.08, settle=0.5, phase="lift")
        P1 = piece_points(objs, recs, range(len(objs)))
        mv = [float((P1[i_].mean(axis=0) - P0[i_].mean(axis=0)) @ sdir) for i_ in range(len(objs))]
        hgt = [float(P1[i_][:, 2].max() - P1[i_][:, 2].min()) for i_ in range(len(objs))]
        gap = []
        for i_ in tgt:  # 몸통과 떨어진 거리(가장 가까운 점 사이, 어림: 점 3000 개까지)
            A_ = P1[i_][:: max(1, len(P1[i_]) // 1500)]
            B_ = P1[body_][:: max(1, len(P1[body_]) // 3000)]
            gap.append(round(1e3 * float(np.sqrt(((A_[:, None, :] - B_[None, :, :]) ** 2).sum(-1)).min()), 1))
        so = {"targets": [recs[i_]["name"] for i_ in tgt], "pushed_mm": [round(1e3 * mv[i_], 1) for i_ in tgt],
              "height_after_mm": [round(1e3 * hgt[i_], 1) for i_ in tgt], "gap_to_body_mm": gap,
              "body_moved_mm": round(1e3 * float(np.linalg.norm(P1[body_].mean(axis=0) - P0[body_].mean(axis=0))), 2),
              "toppled": int(sum(hgt[i_] < 0.012 for i_ in tgt)), "ticks": len(run.rec)}
        so["success"] = bool(so["toppled"] == len(tgt) or min(gap) >= 5.0)
        R_ = run.rec
        sdir_ = out / "separate"
        sdir_.mkdir(exist_ok=True)
        info_ = {"seed": seed, "look": LOOK.current, "mode": "expert_separate", "task": a.task or f"push the {veg.replace('_', ' ')} slices away from the rest with the knife",
                 "veg": veg, "outcome": so, "food_argv": food_argv}
        np.savez_compressed(sdir_ / f"ep_{seed:05d}.npz", images_front=np.stack([r_["front"] for r_ in R_]),
                            images_wrist=np.stack([r_["wrist"] for r_ in R_]), state=np.stack([r_["state"] for r_ in R_]),
                            cmd=np.stack([r_["cmd"] for r_ in R_]), t=np.array([r_["t"] for r_ in R_]),
                            block=np.zeros(7, np.float32), cmd_steps=np.stack(run.cmd_steps),
                            step_phase=np.array([r_["phase"] for r_ in run.rows]), tick_phase=np.array([r_.get("phase", "") for r_ in R_]), grip_cmd=np.array([r_.get("grip", np.nan) for r_ in R_], np.float32))
        (sdir_ / f"ep_{seed:05d}.json").write_text(json.dumps(info_, indent=1, default=float))
        print(f"[food_robot] separate seed {seed}: " + json.dumps(so, default=float), flush=True)
        import imageio.v2 as imageio
        for n_ in ("front", "wrist"):
            for i_ in np.linspace(0, len(R_) - 1, 6).astype(int):
                (sdir_ / "frames").mkdir(exist_ok=True)
                imageio.imwrite(sdir_ / "frames" / f"ep{seed:05d}_{n_}_{i_:03d}.png", R_[i_][n_])
        run.rec, run.cmd_steps, run.rows = None, None, []
        return so

    def glue_after_cut(objs, recs, wet_, args_):
        """썰기 끝 칼 자세에서 칼에 붙은 조각을 찾아 붙여 둠(--glue on) + 접착 하중 감시(GlueMonitor)를 물리 단계마다 돌림.
        → (찾은 목록, 붙인 목록, 감시기)."""
        import food_handoff as FH
        if a.glue != "on" or not objs:
            return [], [], None
        wet_ = float(wet_ if wet_ is not None else (getattr(args_, "wet", None) or 1.0))
        tau_ = float(getattr(args_, "adh_tau", None) or 30.0)
        mu_ = float(getattr(args_, "knife_friction_post", None) or 0.1)
        stuck = FH.knife_stuck_pieces(recs, run.knife_pose(), half_len=knife_geo.half_length, height=knife_geo.height,
                                      spacing=float(g0["spacing"]), z_board=ZB, wet=wet_, tau=tau_, p_a=250.0, mu=mu_,
                                      edge=knife_geo.edge_thickness, spine=knife_geo.spine_thickness, bevel=knife_geo.bevel_height)
        if a.glue_model == "slide":  # 11-25 밤 reports/78: 미끄러지는 D6 이음(관성·손가락·벽·충격을 PhysX 가 같은 식으로)
            glued = FH.glue_slide_to_knife(og, knife, objs, stuck, f_n=a.slide_fn, edge=knife_geo.edge_thickness,
                                           spine=knife_geo.spine_thickness, bevel=knife_geo.bevel_height)
            hl_ = float(knife_geo.half_length)
            mon = FH.SlideMonitor(og, knife, objs, glued, dt, face=dict(x0=-hl_, x1=hl_, z0=0.0, top_fn=knife_geo.spine_height)) if glued else None
        else:
            glued = FH.glue_to_knife(og, knife, objs, stuck)
            mon = FH.GlueMonitor(og, knife, glued, dt) if glued else None
        run.post_step = mon.update if mon is not None else None
        print(f"[food_robot] 칼에 붙은 조각 {len(stuck)}개, 붙여 둠 {len(glued)}개: "
              + "; ".join(f"{s_['name']} 넓이 {s_['area_mm2']:.0f} mm² 버팀 {1e3 * s_['break_N']:.1f} mN(무게 {1e3 * s_['weight_N']:.1f})"
                          + ("" if s_["glue"] else " 안 붙임") for s_ in stuck), flush=True)
        return stuck, glued, mon

    for _ in range(20):
        run.step(arm.eef(), T.OPEN)
    if a.task_kind in ("move_replay", "unstick_replay"):
        # 11-25 reports/72: 썰기 없이 저장한 '잘린 뒤 강체 조각'을 다시 놓고 옮기기만(조각 고르기 시험).
        # 11-25 reports/77 unstick_replay: 칼을 쥐고 그 판의 썰기 끝 칼 자세(마지막 명령 × 칼–손)로 간 뒤 조각을 놓고, 칼에 붙어 있던 조각을 붙여(--glue)
        #   털기(do_unstick) → 칼 놓기 → 옮기기(do_move). 칼 쥐기는 지금 규칙(canonical) — 3판 이후 판과 칼–손 관계가 같음.
        # (re 는 모듈 맨 위에서 불러옴 — 여기서 `import re` 하면 main 전체에서 re 가 지역 이름이 되어 썰기 쪽 집합 내포가 NameError, 11-25 07:30)
        import food_handoff as FH
        unstick_mode = a.task_kind == "unstick_replay"
        park = dict(position=th.tensor([0.32, -0.28, TABLE_H + 0.02], dtype=th.float32),
                    orientation=th.tensor([0.7071068, 0.0, 0.0, 0.7071068], dtype=th.float32))
        knife.set_position_orientation(**park)
        knife.root_link.enable_gravity()
        veg = g0["args"].veg
        (out / "setup.json").write_text(json.dumps({"task_kind": a.task_kind, "robot_pos": robot_pos.tolist(), "args": vars(a),
                                                    "food_argv": food_argv}, indent=1, default=str))
        log = open(out / "episodes.jsonl", "a")
        for e, rd in enumerate(a.replay or []):
            rd = Path(rd).resolve()
            rj = json.loads((rd / "rigid_pieces.json").read_text())
            rz_ = np.load(rd / "rigid_pieces.npz")
            recs = [dict(p_, local=rz_[f"{p_['name']}_local"], particles=rz_[f"{p_['name']}_particles"]) for p_ in rj["pieces"]]
            m_ = re.search(r"ep_(\d+)_phys", str(rd))
            seed = int(m_.group(1)) if m_ else e
            t_ep = time.time()
            uo, so, wet_r, glued = None, None, None, []
            if unstick_mode:
                ep_j = json.loads((rd.parent / f"ep_{seed:05d}.json").read_text())
                ep_z = np.load(rd.parent / f"ep_{seed:05d}.npz", allow_pickle=True)
                K_end = np.asarray(ep_z["cmd_steps"][-1], float) @ np.asarray(ep_j["knife_in_eef"], float)
                wet_r = ep_j["outcome"].get("wet")
                for _ in range(10):
                    run.step(arm.eef(), T.OPEN)
                grasp_r = grasp_knife_geo(T, run, knife, robot, arm, knife_geo, canonical=a.knife_grasp == "canonical")
                K_in_E_r = np.asarray(grasp_r["knife_in_eef"])
                run.K_in_E = K_in_E_r
                K_up = K_end.copy()
                K_up[2, 3] += 0.10
                T.move_knife(run, K_in_E_r, K_up, T.CLOSE, speed=0.15, settle=0.2, phase="replay_up")
                T.move_knife(run, K_in_E_r, K_end, T.CLOSE, speed=0.06, settle=0.4, phase="replay_down")
            objs = FH.add_rigid(og, env.scene, [{"name": f"r{e}_{p_['name']}", "usd": p_["usd"], "category": "food",
                                                 "pos": p_["centre_w"], "quat": [0, 0, 0, 1]} for p_ in recs])
            if unstick_mode:
                K_cut_r = run.knife_pose()
                _, glued, mon = glue_after_cut(objs, recs, wet_r, g0["args"])  # 칼에 붙어 있던 조각을 바로 붙임(기다리면 떨어짐)
                if os.environ.get("GLUE_DEBUG"):
                    print("[glue] 붙인 직후", FH.glue_state(knife, objs, glued), flush=True)
                for i_ in range(int(round(0.2 / dt))):
                    run.step(K_end @ np.linalg.inv(K_in_E_r), T.CLOSE, K_cmd=K_end)
                    if os.environ.get("GLUE_DEBUG") and i_ % 4 == 0:
                        print(f"[glue] {i_ * dt:.3f}s", [(s_["drift_mm"], s_["attached"]) for s_ in FH.glue_state(knife, objs, glued)], flush=True)
                uo = do_unstick(objs, recs, glued, seed, e, veg, K_in_E_r, mon) if a.unstick != "off" else None
                so = None
                if a.separate == "push" and (mon is None or len(mon.broken) == len(glued)):
                    so = do_separate(objs, recs, seed, e, veg, K_in_E_r, K_cut_r)
                if glued:
                    FH.unglue(og, glued)
                run.post_step = None
                for _ in range(40):  # 손을 열어 칼을 놓고 치움(기록 밖)
                    run.step(arm.eef(), T.OPEN)
                knife.set_position_orientation(**park)
                knife.set_linear_velocity(th.zeros(3))
                knife.set_angular_velocity(th.zeros(3))
                knife.root_link.enable_gravity()
            for _ in range(int(round(1.0 / dt))):  # 자리 잡기(썰기 판도 강체로 바꾼 뒤 1 초 둠)
                run.step(arm.eef(), T.OPEN)
            mo = do_move(objs, recs, seed, e, veg)
            info = {"seed": seed, "replay_from": str(rd), "veg": veg, "unstick": uo, "separate": so, "outcome": mo, "wall_s": round(time.time() - t_ep, 1)}
            log.write(json.dumps(info, default=float) + "\n")
            log.flush()
            og.sim.batch_remove_objects(objs)
            for _ in range(int(round(0.5 / dt))):
                run.step(arm.eef(), T.OPEN)
        log.close()
        print(f"[food_robot] {'털기+옮기기' if unstick_mode else '옮기기'} 다시 하기 {len(a.replay or [])} 판 {time.time() - t_wall:.0f} s", flush=True)
        os._exit(0)
    if a.task_kind == "place":  # 단계 C 과제 ②: 빈 손으로 통 식재료(강체)를 탁자에서 집어 도마 위 정한 자리에 놓기
        import food_handoff as FH
        ff = finger_frame(T, robot, arm)
        x_rest = np.asarray(g0["X_REST0"], float)  # 우리 좌표(받침 면 z 0) 입자 틀
        veg = g0["args"].veg
        food_usd, c_food, m_food = FH.food_rigid_usd(out / "food" / f"{veg}.usd", f"{veg}_whole", x_rest,
                                                      float(g0["sim"].particle_mass), np.array(g0["V"]["skin"], float), category=veg)
        size = np.ptp(x_rest, axis=0)
        print(f"[food_robot] 놓기: 통 {veg} 강체 {1e3 * m_food:.1f} g, 크기 {np.round(1e3 * size, 1).tolist()} mm, 무게중심(우리 좌표) "
              f"{np.round(1e3 * c_food, 1).tolist()} mm", flush=True)
        (out / "setup.json").write_text(json.dumps({"task_kind": "place", "robot_pos": robot_pos.tolist(), "args": vars(a), "food_argv": food_argv,
                                                    "food_usd": food_usd, "food_mass_g": 1e3 * m_food, "finger_frame": [v.tolist() for v in ff]},
                                                   indent=1, default=str))
        log = open(out / "episodes.jsonl", "a")
        food_obj = None
        for e in range(a.episodes):
            seed = a.seed0 + e
            look_ep = look_for(seed)  # 11-26 reports/81: 화면 겉모습(base 면 아무것도 안 함)
            rng = np.random.default_rng(seed)
            t_ep = time.time()
            p0 = np.array([rng.uniform(0.20, 0.28), rng.uniform(0.14, 0.22)])  # 탁자 위, 도마(±0.17×±0.12) 밖, 로봇 쪽
            yaw0 = rng.uniform(-math.pi / 6, math.pi / 6)
            p1 = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)])  # 도마 위 놓을 자리
            yaw1 = rng.uniform(-math.pi / 9, math.pi / 9)
            pos0 = np.array([p0[0], p0[1], TABLE_H + 0.002]) + rz(yaw0) @ c_food
            q0 = [0.0, 0.0, math.sin(0.5 * yaw0), math.cos(0.5 * yaw0)]
            if food_obj is None:
                food_obj = FH.add_rigid(og, env.scene, [{"name": "food_whole", "usd": food_usd, "category": veg, "pos": pos0, "quat": q0}])[0]
                if a.food_view == "mesh":  # 11-25 B5: 통 식재료 메쉬(칼질로 안 나눔)를 강체에 붙이고 볼록 껍질 그림은 숨김
                    mdir, t_prep = prep_mesh(x_rest, food_meta(g0, g0["spacing"]), mesh_cache, a.mesh_tex, True, a.prep_python)
                    v = new_view("food_whole_mesh", mdir)
                    v.bind_rigid_local(food_obj, {i_: pc_["V0"] - c_food for i_, pc_ in enumerate(v.pieces)})
                    FMV.hide_rigid_visuals(og.sim.stage, [food_obj])
                    v.refresh()
                    for _ in range(40):  # 질감·새 메쉬가 올라오게
                        og.sim.render()
                    print(f"[food_robot] 메쉬 식재료: {mdir.name} 만들기 {t_prep:.1f} s, {v.stats()}", flush=True)
            else:
                food_obj.set_position_orientation(position=th.tensor(pos0, dtype=th.float32), orientation=th.tensor(q0, dtype=th.float32))
                food_obj.set_linear_velocity(th.zeros(3))
                food_obj.set_angular_velocity(th.zeros(3))
            E_up = arm.eef()
            E_up[:3, 3] += np.array([0.0, 0.0, 0.0])
            for _ in range(int(round(0.6 / dt))):  # 자리 잡기(기록 밖)
                run.step(E_up, T.OPEN)
            fp, fq = (T.np_(v) for v in food_obj.get_position_orientation())
            yaw_f = FH.yaw_of(fq)
            close0 = rz(yaw_f) @ np.array([0.0, 1.0, 0.0])  # 짧은 축(우리 y)으로 닫음
            grasp_c = fp + np.array([0.0, 0.0, 0.003])
            above0 = topdown_eef(grasp_c + np.array([0.0, 0.0, 0.12]), close0, ff)
            move_hand(T, run, above0, T.OPEN, speed=0.25, settle=0.3, phase="home")  # 시작 자세(기록 밖)
            for _ in range(15):  # DLSS 가 앞 장면을 섞어 그리지 않게(팀원 warmup_renders)
                og.sim.render()
            run.rec, run.cmd_steps, run.rows = [], [], []
            move_hand(T, run, topdown_eef(grasp_c + np.array([0.0, 0.0, 0.03]), close0, ff), T.OPEN, speed=0.12, settle=0.1, phase="approach")
            move_hand(T, run, topdown_eef(grasp_c, close0, ff), T.OPEN, speed=0.05, settle=0.2, phase="descend")
            for _ in range(int(round(0.6 / dt))):
                run.phase = "close"
                run.step(topdown_eef(grasp_c, close0, ff), T.CLOSE)
            held0 = robot.is_grasping(arm.arm, candidate_obj=food_obj) == run.grasp_true
            lift = topdown_eef(grasp_c + np.array([0.0, 0.0, 0.10]), close0, ff)
            move_hand(T, run, lift, T.CLOSE, speed=0.10, settle=0.1, phase="lift")
            tgt_c = np.array([p1[0], p1[1], ZB + 0.002]) + rz(yaw1) @ c_food + np.array([0.0, 0.0, 0.003])
            close1 = rz(yaw1) @ np.array([0.0, 1.0, 0.0])
            if close1 @ close0 < 0:  # 손을 덜 돌리게(짧은 축은 앞뒤가 같음)
                close1 = -close1
            move_hand(T, run, topdown_eef(tgt_c + np.array([0.0, 0.0, 0.10]), close1, ff), T.CLOSE, speed=0.15, settle=0.2, phase="carry")
            move_hand(T, run, topdown_eef(tgt_c + np.array([0.0, 0.0, 0.004]), close1, ff), T.CLOSE, speed=0.05, settle=0.3, phase="lower")
            for _ in range(int(round(0.5 / dt))):
                run.phase = "release"
                run.step(topdown_eef(tgt_c + np.array([0.0, 0.0, 0.004]), close1, ff), T.OPEN)
            move_hand(T, run, topdown_eef(tgt_c + np.array([0.0, 0.0, 0.12]), close1, ff), T.OPEN, speed=0.10, settle=0.5, phase="retreat")
            fp1, fq1 = (T.np_(v) for v in food_obj.get_position_orientation())
            yaw_e = (FH.yaw_of(fq1) - yaw1 + math.pi / 2) % math.pi - math.pi / 2  # 앞뒤 같은 모양: 180° 차이는 같은 놓임
            want = np.array([p1[0], p1[1], ZB + 0.002]) + rz(yaw1) @ c_food
            xy_err = 1e3 * float(np.linalg.norm(fp1[:2] - want[:2]))
            z_err = 1e3 * float(fp1[2] - (ZB + c_food[2]))
            held_end = robot.is_grasping(arm.arm, candidate_obj=food_obj) == run.grasp_true
            o = {"held_after_close": bool(held0), "held_at_end": bool(held_end), "xy_err_mm": round(xy_err, 2), "z_err_mm": round(z_err, 2),
                 "yaw_err_deg": round(math.degrees(yaw_e), 2), "start_xy": p0.tolist(), "start_yaw_deg": math.degrees(yaw0),
                 "target_xy": p1.tolist(), "target_yaw_deg": math.degrees(yaw1), "ticks": len(run.rec), "wall_s": round(time.time() - t_ep, 1),
                 "pos_err_mm_max": round(float(max((r_.get("pos_err_mm", 0.0) for r_ in run.rows), default=0.0)), 3)}
            o["success"] = bool(held0 and not held_end and xy_err < 15.0 and abs(z_err) < 8.0 and abs(math.degrees(yaw_e)) < 15.0)
            o["food_view"] = run.food_view.stats() if run.food_view is not None else "points" if a.food_view == "points" else None
            task = a.task or f"put the {veg.replace('_', ' ')} on the cutting board"
            R = run.rec
            blk = np.array([*fp1, FH.yaw_of(fq1), *size], np.float32)
            info = {"seed": seed, "mode": "expert_place", "task": task, "veg": veg, "outcome": o, "food_argv": food_argv, "look": look_ep}
            np.savez_compressed(out / f"ep_{seed:05d}.npz", images_front=np.stack([r_["front"] for r_ in R]),
                                images_wrist=np.stack([r_["wrist"] for r_ in R]), state=np.stack([r_["state"] for r_ in R]),
                                cmd=np.stack([r_["cmd"] for r_ in R]), t=np.array([r_["t"] for r_ in R]), block=blk,
                                cmd_steps=np.stack(run.cmd_steps), step_phase=np.array([r_["phase"] for r_ in run.rows]),
                                tick_phase=np.array([r_.get("phase", "") for r_ in R]), grip_cmd=np.array([r_.get("grip", np.nan) for r_ in R], np.float32))
            (out / f"ep_{seed:05d}.json").write_text(json.dumps(info, indent=1, default=float))
            log.write(json.dumps(info, default=float) + "\n")
            log.flush()
            print(f"[food_robot] place seed {seed}: " + json.dumps(o), flush=True)
            if e < 2:
                import imageio.v2 as imageio
                for n_ in ("front", "wrist"):
                    for i_ in np.linspace(0, len(R) - 1, 6).astype(int):
                        (out / "frames").mkdir(exist_ok=True)
                        imageio.imwrite(out / "frames" / f"ep{seed:05d}_{n_}_{i_:03d}.png", R[i_][n_])
            run.rec, run.cmd_steps, run.rows = None, None, []
        log.close()
        print(f"[food_robot] 놓기 {a.episodes} 에피소드 {time.time() - t_wall:.0f} s", flush=True)
        os._exit(0)
    grasp = grasp_knife_geo(T, run, knife, robot, arm, knife_geo, canonical=a.knife_grasp == "canonical")
    K_in_E = np.asarray(grasp["knife_in_eef"])
    run.K_in_E = K_in_E
    E_in_K = np.linalg.inv(K_in_E)
    print("[food_robot] 칼 잡기:", json.dumps({k: v for k, v in grasp.items() if k != "knife_in_eef"}), flush=True)
    if not grasp["held"]:  # 11-24 r2: 진단 — assisted 잡기 = 손가락 둘 다 닿음(접촉 보고) ∩ 손끝 사이 광선이 맞힘
        try:
            cand, links_ = robot._find_gripper_contacts(arm=arm.arm)
            rays = robot._find_gripper_raycast_collisions(arm=arm.arm)
            print("[food_robot] 잡기 진단: 접촉 후보", sorted(cand), "| 접촉 링크", {k_: sorted(v_) for k_, v_ in links_.items()},
                  "| 광선 후보", sorted(rays), "| 손가락", [l_.prim_path for l_ in robot.finger_links[arm.arm]],
                  "| 칼", knife.prim_path, list(knife.links), "| 칼 위치", T.np_(knife.get_position_orientation()[0]).round(4).tolist(),
                  "| 손끝", T.np_(arm.eef()[:3, 3]).round(4).tolist(), flush=True)
        except Exception as _e:  # noqa: BLE001
            print("[food_robot] 잡기 진단 실패:", _e, flush=True)
        raise SystemExit("집게가 칼을 못 쥠")
    setup_rec = {"robot_pos": robot_pos.tolist(), "robot_yaw_deg": math.degrees(robot_yaw), "grasp": grasp, "args": vars(a),
                 "food_argv": food_argv, "team_code": str(Path(T.__file__).resolve()), "solver": str(Path(simfoundry.__file__).resolve().parent)}
    (out / "setup.json").write_text(json.dumps(setup_rec, indent=1, default=str))
    log = open(out / "episodes.jsonl", "a")
    g = g0
    for e in range(a.episodes):
        seed = a.seed0 + e
        look_ep = look_for(seed)  # 11-26 reports/81: 화면 겉모습(base 면 아무것도 안 함, 옮기기·떼기도 같은 겉모습)
        t_ep = time.time()
        if e > 0:
            g, wet = setup(seed)
        else:
            wet = wet0
        t_setup = time.time() - t_ep
        tm0 = (run.t_cap, run.n_cap, run.t_wide, run.n_wide)
        args_f, sim, fixes, off = g["args"], g["sim"], g["fixes"], g["off"]
        rng = np.random.default_rng(seed)
        shift = rng.uniform(-a.shift, a.shift, 2) if a.shift > 0 else np.zeros(2)
        Woff = np.asarray(g["W_OFF"], float) + np.array([shift[0], shift[1], 0.0])
        yaw_w = math.radians(rng.uniform(-a.yaw_deg, a.yaw_deg)) if a.yaw_deg > 0 else 0.0  # 식재료(와 칼 경로)를 도마 위에서 돌림
        Wm = np.eye(4)
        Wm[:3, :3] = rz(yaw_w)
        Wm[:3, 3] = Woff  # 우리 좌표 → 월드
        S = np.eye(4)
        S[:3, 3] = np.asarray(off, float)
        S = S @ np.linalg.inv(Wm)  # 월드 → 솔버(= 우리 좌표 + off)
        kp_our = g["knife_pose_our"]

        def K_world(q):
            return T.knife_world(kp_our(q), Wm)

        # 경로 시작 자리 위로 → 시작 자리(기록 안 함)
        start = K_world(g["Qg"][0])
        above = start.copy()
        above[2, 3] += 0.06
        T.move_knife(run, K_in_E, above, T.CLOSE, ang_speed=a.approach_ang_speed, settle=0.3, phase="home")
        T.move_knife(run, K_in_E, start, T.CLOSE, speed=0.05, ang_speed=a.approach_ang_speed, settle=0.3, phase="home")
        coupling = PlanCoupling(sim, S, settle=0.0, linger=1.0, auto_cut=False)
        coupler = Coupler(coupling, knife.root_link.prim_path, apply_wrench=True)
        V, spacing, R0 = g["V"], g["spacing"], g["R0"]
        x0o = sim.x0 - off
        rr = np.sqrt(x0o[:, 1] ** 2 + (x0o[:, 2] - R0) ** 2) / max(R0, 1e-9)
        fi, fo = np.array(V["flesh_in"]), np.array(V["flesh_out"])
        albedo = np.where(sim.skin[:, None], np.array(V["skin"]), fi[None] * (1 - rr[:, None]) + fo[None] * rr[:, None])
        if run.points is not None:
            run.points.points.GetPrim().SetActive(False)
            run.points = None
        run.world_from_sim = coupling.world_from_sim()
        t_prep, n_warm = 0.0, 15
        if a.food_view == "mesh":  # 11-25 B5: 계획된 칼질 평면으로 미리 나눈 메쉬가 입자를 따라감
            x_mesh = (g["X_REST0"] if g.get("PLAN2") else (sim.x0 - off))
            mdir, t_prep = prep_mesh(x_mesh, food_meta(g, spacing), mesh_cache, a.mesh_tex, False, a.prep_python)
            v = new_view(f"food_mesh_{e}", mdir)
            Wfs = np.array(run.world_from_sim, float)
            v.follow_particles(lambda sim=sim, W=Wfs: sim.positions() @ W[:3, :3].T + W[:3, 3], labels=lambda sim=sim: sim.pieces())
            v.refresh()
            n_warm = 40 if t_prep > 0 or e == 0 else 15  # 새 질감이면 올라올 때까지 더 그림
            print(f"[food_robot] 메쉬 식재료: {mdir.name} 만들기 {t_prep:.1f} s, {v.stats()}", flush=True)
        else:
            run.points = FoodPoints(og.sim.stage, f"/World/food_{e}", sim.skin, 1.6 * spacing, None, None, albedo=albedo)
            run.points.update(sim.positions() @ run.world_from_sim[:3, :3].T + run.world_from_sim[:3, 3])
        run.coupler = coupler
        for _ in range(n_warm):  # DLSS 가 새 입자·메쉬를 보이게(팀원 warmup_renders)
            og.sim.render()
        cap = g["FORCE_CAP_N"]
        gate = (ForceGate(cap, mode=args_f.force_mode, band=args_f.force_band,
                          filter_steps=max(1, int(round(args_f.force_filter_ms * 1e-3 / dt))), stall_s=args_f.stall_s,
                          stall_dz=args_f.stall_dz_mm * 1e-3, stall_max_s=args_f.stall_max_s, step_dt=dt)
                if cap is not None else None)
        fol = PlanFollower(g, gate, dt)
        if a.track_mm and a.track_mm[1] > 0:
            fol.track = (a.track_mm[0] * 1e-3, a.track_mm[1] * 1e-3)
            fol.track_floor = float(a.track_floor)
        PH, plan_cuts, PLAN2 = g["PH"], g.get("plan_cuts"), g.get("PLAN2")
        settle_end, follow_steps = g["settle_end"], g["follow_steps"]
        settle2_end, follow2 = g["settle2_end"], g["follow2"]
        kf_cut = float(sim.material.knife_friction)
        frames = {"x": [], "lab": [], "step": [], "t": [], "q": [], "q_cmd": [], "pr": []}
        adh = []
        coupler.start()
        run.rec, run.cmd_steps, run.rows = [], [], []
        if a.overview:
            run.overview_every = max(1, round(hz / a.overview_fps))
            run.overview_dir = out / "overview_frames" / f"ep{seed:05d}"
            run.overview_dir.mkdir(parents=True, exist_ok=True)
            run.overview_n = 0
            if a.video_all:  # 11-26 reports/79: 썰기는 식재료(와 채썰기 쌓을 자리) 가까이
                xr_ = np.asarray(g["X_REST0"], float)
                x_hi = float(PLAN2["x_end"]) if PLAN2 else float(xr_[:, 0].max())
                tgt_our = np.array([0.5 * (float(xr_[:, 0].min()) + x_hi), 0.0, 0.012, 1.0])
                aim_overview((Wm @ tgt_our)[:3], [0.10, -0.30, 0.20])
                vmark("cut_start")
        import dataclasses
        k_phys = 0
        fz = fy = 0.0
        knife_zmin = {}
        hold_log = []  # (경로 줄, 때, 손이 잡은 입자 수)
        while not fol.done and fol.t_ms < a.max_ep_s * 1e3:
            q_cmd, crossed, r = fol.next()
            if any(j in fol.cut_steps for j in crossed):
                coupling.pending = True
            if PLAN2 and PLAN2["relocate_step"] in crossed:
                n_fr0 = len(g["frames"]["x"])
                vmark("stack_start")
                g["relocate"](PLAN2["relocate_step"])
                if a.video_all and a.overview and run.food_view is not None and run.overview_dir is not None:
                    # 11-26 reports/79: 쌓기는 물리 순간 이동 — 영상에만 조각을 하나씩 들어 눕혀 옮기는 장면(og_veg_cut 이 만든 carry 프레임)을 끼움
                    Wv = np.array(run.world_from_sim, float)
                    for Xc in g["frames"]["x"][n_fr0:]:
                        Pw = (np.asarray(Xc, float) + off) @ Wv[:3, :3].T + Wv[:3, 3]
                        run.food_view.follow_particles(lambda P=Pw: P)
                        run.wide_frame()
                    run.food_view.follow_particles(lambda sim=sim, W=Wv: sim.positions() @ W[:3, :3].T + W[:3, 3], labels=lambda sim=sim: sim.pieces())
                vmark("stack_end")
            if args_f.knife_friction_post is not None:
                ph_now = str(PH[min(int(fol.s), fol.n - 1)])
                kf_want = kf_cut if ph_now.startswith(("cut", "bsaw", "start", "settle")) else args_f.knife_friction_post
                if sim.material.knife_friction != kf_want:
                    sim.material = dataclasses.replace(sim.material, knife_friction=kf_want)
            K_cmd = K_world(q_cmd)
            run.phase = str(PH[min(int(fol.s), fol.n - 1)])
            K = run.step(K_cmd @ E_in_K, T.CLOSE, K_cmd=K_cmd, extra={"row": float(fol.s), "rate": r})
            k_phys += 1
            row = run.rows[-1]
            if coupling.prev is not None:
                _zk = float(np.asarray(coupling.prev.origin)[2] - off[2])
                fol.err_z = _zk - float(q_cmd[2])  # 11-25: 팔 기다리기(PlanFollower.track)에 씀
                if run.phase.startswith(("cut", "dwell")):  # 11-25: 칼질마다 실제 칼(날 끝 기준 자세) 가장 낮은 높이(우리 좌표)
                    _kk = "cut" + run.phase.lstrip("cutdwel")
                    knife_zmin[_kk] = min(knife_zmin.get(_kk, 1.0), _zk)
            if row.get("advanced"):
                fz, fy = float(row.get("fz", 0.0)), float(row.get("fy", 0.0))
            fol.after(crossed, fz, fy)
            if fixes is not None:
                last = any(j + 1 >= fol.n or PH[j + 1] != PH[j] for j in crossed)
                for j in crossed:  # 11-25: og_veg_cut.py 와 같이 손 상자를 정한 뒤 grip() 으로 실제로 잡음(11-24 판은 grip 을 빠뜨려 손이 없었음 → 칼에 밀려 식재료가 4.7 mm 미끄러짐)
                    if j == settle_end:
                        if args_f.hold_follow is not None and plan_cuts:
                            fixes.hold_box = g["hand_box"](plan_cuts[0] - args_f.hold_follow)
                        hold_log.append((int(j), "settle", fixes.grip()))
                    elif args_f.hold_follow is not None and j in follow_steps:
                        fixes.hold_box = g["hand_box"](follow_steps[j] - args_f.hold_follow)
                        hold_log.append((int(j), "follow", fixes.grip()))
                    elif PLAN2 and j == settle2_end:
                        fixes.hold_box = g["hand_box2"](PLAN2["cuts"][0] - args_f.hold_follow2)
                        hold_log.append((int(j), "settle2", fixes.grip()))
                    elif PLAN2 and j in follow2:
                        fixes.hold_box = g["hand_box2"](follow2[j] - args_f.hold_follow2)
                        hold_log.append((int(j), "follow2", fixes.grip()))
                fixes.update(int(fol.t_ms), PH[min(int(fol.s), fol.n - 1)], last, q_cmd)
                if fixes.adhesion is not None:
                    adh.append(fixes.adhesion_read())
            if k_phys % max(1, int(round(0.04 / dt))) == 0:  # 우리 진단·메쉬 렌더용 프레임(0.04 s 쯤마다)
                Kc = coupling.prev
                q_act = np.array([*(np.asarray(Kc.origin) - off), Kc.yaw + 0.5 * math.pi]) if Kc is not None else q_cmd
                frames["x"].append((sim.positions() - off).astype(np.float32))
                frames["lab"].append(sim.pieces().astype(np.int8))
                frames["step"].append(min(int(fol.s), fol.n - 1))
                frames["t"].append(fol.t_ms * 1e-3)
                frames["q"].append(q_act.astype(np.float64))
                frames["q_cmd"].append(np.asarray(q_cmd, np.float64))
                frames["pr"].append([float(Kc.pitch), float(Kc.roll)] if Kc is not None else [np.nan, np.nan])  # 11-25: 칼 기울기(솔버 틀)
            if k_phys % 600 == 0:
                print(f"  t {fol.t_ms * 1e-3:6.2f}s 줄 {fol.s:8.1f}/{fol.n} {run.phase:10s} Fz {fz:6.2f} N 몫 {r:.2f} "
                      f"자세 오차 {row.get('pos_err_mm', float('nan')):.2f} mm 조각 {len(set(sim.pieces().tolist()))} "
                      f"{time.time() - t_ep:.0f}s", flush=True)
        coupler.stop()
        n_cuts = len({str(p_) for p_ in PH if re.match(r"cut\d+$", str(p_))})  # 경로(--max_steps 로 잘렸으면 그 안)에 든 칼질 수
        complete = [bool(sim.cut_complete(c)) for c in range(sim.n_cuts)]
        held = robot.is_grasping(arm.arm, candidate_obj=knife) == run.grasp_true
        cuts = []
        for k_ in sorted(fol.cuts):
            c = fol.cuts[k_]
            cuts.append({"k": k_, "outcome": c["outcome"] or "unfinished", "why": c.get("why"), "slow_s": c["slow"] * dt,
                         "z_min_mm": c["z_min"] * 1e3, "fz_max_N": c["fz_max"], "fz_mean_N": c["fz_sum"] / max(c["n"], 1),
                         "fy_absmax_N": c["fy_absmax"]})
        stalls = sum(c["outcome"] == "stall" for c in cuts)
        rows = run.rows
        o = {"cuts_planned": n_cuts, "cuts_begun": int(sim.n_cuts), "cut_complete": complete, "stalls": int(stalls),
             "held": bool(held), "sim_blown": bool(coupler.blown), "pieces": int(len(set(sim.pieces().tolist()))),
             "max_force_n": round(float(max((math.sqrt(r_.get("fx", 0) ** 2 + r_.get("fy", 0) ** 2 + r_.get("fz", 0) ** 2) for r_ in rows),
                                            default=0.0)), 3),
             "pos_err_mm_max": round(float(max((r_.get("pos_err_mm", 0.0) for r_ in rows), default=0.0)), 3),
             "pos_err_mm_median": round(float(np.median([r_.get("pos_err_mm", 0.0) for r_ in rows])) if rows else 0.0, 3),
             "physics_steps": len(rows), "ticks": len(run.rec), "plan_time_s": round(fol.t_ms * 1e-3, 3),
             "force_cap_N": cap, "wet": wet, "shift_m": shift.tolist(), "yaw_deg": math.degrees(yaw_w), "cuts": cuts,
             "wall_s": round(time.time() - t_ep, 1), "fast3": bool(a.fast3)}
        _phys = float(sum(r_.get("step_wall_s", 0.0) for r_ in rows))
        _mpm = float(sum(r_.get("mpm_wall_s", 0.0) or 0.0 for r_ in rows))
        _cap, _ncap, _wide, _nwide = run.t_cap - tm0[0], run.n_cap - tm0[1], run.t_wide - tm0[2], run.n_wide - tm0[3]
        o["timing_s"] = {"setup": round(t_setup, 1), "physics_step": round(_phys, 1), "of_which_mpm": round(_mpm, 1),
                         "capture": round(_cap, 1), "captures": int(_ncap), "wide_frames": round(_wide, 1), "wide_n": int(_nwide),
                         "other": round(o["wall_s"] - t_setup - _phys - _cap - _wide, 1)}
        sl = slice_report(np.asarray(sim.x0 - off, float), sim.pieces(), [m_["x_mm"] / 1e3 for m_ in g["gmeta"]["marks"]][:n_cuts],
                          float(spacing), g["gmeta"].get("plan_mm"))
        o["slices"] = sl
        o["slices_separated"] = int(sum(r_["separated"] for r_ in sl))
        o["thickness_ok"] = int(sum(bool(r_["thick_ok"]) for r_ in sl))
        o["knife_zmin_mm"] = {k_: round(1e3 * v_, 2) for k_, v_ in sorted(knife_zmin.items())}
        o["hand_grips"] = hold_log
        o["track_hold_s"] = round(fol.track_hold_ms * 1e-3, 3)  # 팔을 기다리느라 경로가 늦어진 시간
        _lab = sim.pieces()
        _body = _lab == np.bincount(_lab).argmax()  # 몸통(가장 큰 조각) 입자만 — 칼에 붙어 들린 편이 평균을 끌지 않게
        o["food_slide_mm"] = np.round(1e3 * np.median(sim.positions()[_body] - np.asarray(sim.x0, float)[_body], axis=0), 2).tolist()  # 11-25: 몸통이 얼마나 밀렸나(손이 잡으면 0 근처)
        _cv, _co = sim.cut_v.numpy(), sim.cut_o.numpy()  # 11-25: 칼질 평면(begin_cut 때 실제 칼 자세로 정해짐) — 계획 평면(법선 +x)과 얼마나 다른가
        o["cut_planes"] = [{"x_mm": round(1e3 * float(_co[c_][0] - off[0]), 3), "tilt_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, float(_cv[c_][2]))))), 3),
                            "yaw_dev_deg": round(math.degrees(math.atan2(float(_cv[c_][1]), float(_cv[c_][0]))), 3)} for c_ in range(sim.n_cuts)]
        o["cut_complete_note"] = "cut_complete = 솔버가 에피소드 끝에 본 값(칼질 평면에 남은 재료가 없으면 참 — 11-25 오이 판에서 거꾸로 나옴, 성공 판정에 안 씀)"
        # 11-25: 성공 = 편마다 몸통과 다른 라벨(솔버가 따로 떨어진 조각으로 앎) + 두께 ±2 mm(교수님 기준) + 칼 쥠·결합 안 끊김·멈춤 0
        o["success"] = bool(held and not coupler.blown and stalls == 0 and sim.n_cuts == n_cuts and len(sl) == n_cuts
                            and all(r_["separated"] and r_["thick_ok"] for r_ in sl))
        o["food_view"] = (dict(run.food_view.stats(), prep_s=round(t_prep, 2)) if run.food_view is not None else a.food_view)
        th_list = g["gmeta"].get("plan_mm")
        task = a.task or f"slice the {args_f.veg.replace('_', ' ')} into {th_list[0]:g} mm slices"
        blk = np.array([*(Woff + rz(yaw_w) @ np.array([(g['X0'] + g['X1']) / 2, (g['Y0'] + g['Y1']) / 2, g['ZT'] / 2])), yaw_w,
                        g["X1"] - g["X0"], g["Y1"] - g["Y0"], g["ZT"]], np.float32)
        R = run.rec
        info = {"seed": seed, "mode": "expert_plan", "look": look_ep, "task": task, "veg": args_f.veg, "plan_mm": th_list, "plan2_mm": g["gmeta"].get("plan2_mm"),
                "outcome": o, "knife_in_eef": K_in_E.tolist(), "food_argv": food_argv}
        np.savez_compressed(out / f"ep_{seed:05d}.npz", images_front=np.stack([r_["front"] for r_ in R]),
                            images_wrist=np.stack([r_["wrist"] for r_ in R]), state=np.stack([r_["state"] for r_ in R]),
                            cmd=np.stack([r_["cmd"] for r_ in R]), t=np.array([r_["t"] for r_ in R]), block=blk,
                            cmd_steps=np.stack(run.cmd_steps),
                            steps=np.array([[r_["t"], r_.get("fx", 0.0), r_.get("fy", 0.0), r_.get("fz", 0.0), np.nan,
                                             r_.get("pos_err_mm", np.nan), np.nan, np.nan, np.nan, np.nan] for r_ in rows], np.float32),
                            step_phase=np.array([r_["phase"] for r_ in rows]), tick_phase=np.array([r_.get("phase", "") for r_ in R]), grip_cmd=np.array([r_.get("grip", np.nan) for r_ in R], np.float32),
                            tick_cuts_done=np.array([r_.get("cuts_done", 0) for r_ in R], np.int16),
                            step_rate=np.array([r_.get("rate", 1.0) for r_ in rows], np.float32),
                            step_row=np.array([r_.get("row", 0.0) for r_ in rows], np.float32))
        (out / f"ep_{seed:05d}.json").write_text(json.dumps(info, indent=1, default=float))
        # 우리 진단·메쉬 렌더용(og_veg_cut.py 의 frames.npz 와 같은 열 + q_cmd)
        pd = out / f"ep_{seed:05d}_phys"
        pd.mkdir(parents=True, exist_ok=True)
        meta = dict(g["gmeta"])
        meta.update({"solver": "teammate Warp MPM in OmniGibson process (food_robot_episodes.py, Franka holds the knife)",
                     "particle_size": spacing, "dx": args_f.dx, "veg": args_f.veg, "material": dataclasses.asdict(g["mat"]),
                     "fixes": fixes.summary() if fixes else None, "world_offset": Woff.tolist()})
        SH = g.get("SH")
        if SH:
            meta.update({"shape": args_f.shape, "food_mesh": SH["source"], "food_xform": SH["xform"], "pre_trims": SH["pre_planes"],
                         "pre_splits": [], "color_in_source_frame": bool(SH.get("stack", False)), "bounds": SH["bounds"]})
            if SH.get("colors"):
                meta.update({"colors": SH["colors"], "category": SH.get("veg", args_f.veg), "layers": SH.get("layers"),
                             "shell": SH.get("shell")})
        x_rest = (g["X_REST0"] if PLAN2 else (sim.x0 - off)).astype(np.float32)
        st = np.array(frames["step"], np.int64)
        np.savez_compressed(pd / "frames.npz", x=np.array(frames["x"]), lab=np.array(frames["lab"]), step=st, t=np.array(frames["t"]),
                            q=np.array(frames["q"]), q_cmd=np.array(frames["q_cmd"]), pr=np.array(frames["pr"], float), phase=PH[st], x_rest=x_rest,
                            mat=np.zeros(len(x_rest), np.int8), meta=json.dumps(meta, ensure_ascii=False, default=float))
        np.savez_compressed(pd / "log.npz", q=g["Qg"], phase=PH, adh_og=np.array(adh, float) if adh else np.zeros((0, 2)))
        (pd / "metrics.json").write_text(json.dumps({"veg": args_f.veg, "outcome": o, "n_particles": int(sim.n_particles),
                                                     "dt_s": float(sim.dt), "physics_hz": hz, "warnings": sim.warnings,
                                                     "fixes": fixes.summary() if fixes else None}, indent=1, default=float))
        log.write(json.dumps(info, default=float) + "\n")
        log.flush()
        print(f"[food_robot] seed {seed}: " + json.dumps({k: o[k] for k in ("success", "cut_complete", "stalls", "held", "pieces",
                                                                          "max_force_n", "pos_err_mm_median", "ticks", "wall_s")}),
              flush=True)
        if e < 2:
            import imageio.v2 as imageio
            for n_ in ("front", "wrist"):
                for i_ in np.linspace(0, len(R) - 1, 6).astype(int):
                    (out / "frames").mkdir(exist_ok=True)
                    imageio.imwrite(out / "frames" / f"ep{seed:05d}_{n_}_{i_:03d}.png", R[i_][n_])
        objs, recs, glued, mon, K_conv = [], [], [], None, None
        if a.to_rigid:  # 단계 C1: 썬 조각 → PhysX 강체(조각 옮기기 과제의 시작 상태)
            import food_handoff as FH
            if run.points is not None:
                run.points.points.GetPrim().SetActive(False)
                run.points = None  # 강체가 된 뒤에는 입자를 그리지 않음
            t_c = time.time()
            vmark("cut_end")
            objs, recs, small, groups_p = FH.pieces_to_rigid(og, env.scene, sim, coupling.world_from_sim(), pd / "pieces", albedo=albedo,
                                                             prefix=f"ep{seed}_piece",
                                                             spacing=float(spacing))  # 11-25: 묶음마다 이어진 덩어리로 나눔(칼에 붙은 편)
            if run.food_view is not None and objs:  # 11-25 B5: 메쉬 조각을 강체에 붙이고 볼록 껍질 그림은 숨김
                bound = run.food_view.bind_rigid(objs, groups_p, [r_["group"] for r_ in recs])
                FMV.hide_rigid_visuals(og.sim.stage, objs)
                print(f"[food_robot] 메쉬 조각 → 강체: {bound} (−1 = 강체 안 된 작은 묶음이라 숨김)", flush=True)
            t_c = time.time() - t_c
            K_conv = run.knife_pose()
            stuck, glued, mon = glue_after_cut(objs, recs, wet, args_f)  # 11-25 reports/77: 칼에 붙은 편은 강체로도 붙여 둠
            unstick_rec = None
            if glued and a.unstick != "off" and a.task_kind == "move":
                unstick_rec = do_unstick(objs, recs, glued, seed, e, args_f.veg, K_in_E, mon)  # 과제 ④: 접시 위에서 털기(기록)
            else:
                up = run.knife_pose()
                up[2, 3] += 0.10  # 칼을 조각 위에서 비킴(기록 밖)
                T.move_knife(run, K_in_E, up, T.CLOSE, speed=0.10, settle=0.0, phase="away")
            p0 = [T.np_(o.get_position_orientation()[0]) for o in objs]
            for _ in range(int(round(1.0 / dt))):
                run.step(run.cmd, T.CLOSE)
            p1 = [T.np_(o.get_position_orientation()[0]) for o in objs]
            moved = [1e3 * float(np.linalg.norm(b - a_)) for a_, b in zip(p0, p1)]
            board_top = TABLE_H + BOARD_T
            below = [1e3 * float(board_top - T.np_(o.aabb[0])[2]) for o in objs]  # 도마 면 아래로 들어간 깊이(+ 면 뚫음)
            rig = {"convert_s": round(t_c, 2), "pieces": [{k_: v for k_, v in r_.items() if k_ not in ("particles", "local")} for r_ in recs],
                   "small_groups_left_out": small, "moved_1s_mm": [round(v, 2) for v in moved],
                   "aabb_below_board_mm": [round(v, 2) for v in below],
                   "knife_pose_w": K_conv.tolist(), "wet": wet, "stuck": stuck, "glued": [gl["name"] for gl in glued],
                   "unstick": unstick_rec}
            (pd / "rigid_pieces.json").write_text(json.dumps(rig, indent=1, default=float))
            np.savez_compressed(pd / "rigid_pieces.npz", **{f"{r_['name']}_particles": r_["particles"] for r_ in recs},
                                **{f"{r_['name']}_local": r_["local"] for r_ in recs})
            print(f"[food_robot] 강체로: 조각 {len(objs)}개({t_c:.1f} s), 1 초 동안 움직임 최대 {max(moved, default=0):.2f} mm, "
                  f"도마 아래 최대 {max(below, default=0):.2f} mm, 작은 묶음 뺌 {len(small)}", flush=True)
            if a.overview and "overview" in cams:
                import imageio.v2 as imageio
                og.sim.render()
                imageio.imwrite(pd / "rigid_overview.png", T.np_(cams["overview"].get_obs()[0]["rgb"])[:, :, :3].astype(np.uint8))
                imageio.imwrite(pd / "rigid_front.png", T.np_(cams["front"].get_obs()[0]["rgb"])[:, :, :3].astype(np.uint8))
        if a.task_kind == "move" and a.to_rigid and objs:  # 과제 ③: 조각 하나를 접시로(칼은 기록 밖에서 치움)
            run.rec, run.cmd_steps = None, None
            sep_rec = None
            if a.separate == "push" and (mon is None or len(mon.broken) == len(glued)):  # 붙은 편이 다 떨어졌을 때만(칼 틈으로 내려가므로)
                sep_rec = do_separate(objs, recs, seed, e, args_f.veg, K_in_E, K_conv)
            if glued:  # 11-25: 아직 붙은 편이 있으면 접착을 끔(칼을 순간 이동하면 이음목이 편을 홱 끌어당김)
                FH.unglue(og, glued)
            run.post_step = None
            for _ in range(40):  # 손을 열어 칼을 놓고
                run.step(arm.eef(), T.OPEN)
            park_q = th.tensor([0.7071068, 0.0, 0.0, 0.7071068], dtype=th.float32)
            knife.set_position_orientation(position=th.tensor([0.32, -0.28, TABLE_H + 0.02], dtype=th.float32), orientation=park_q)
            knife.set_linear_velocity(th.zeros(3))
            knife.set_angular_velocity(th.zeros(3))
            knife.root_link.enable_gravity()
            if a.video_all and a.overview:  # 11-26 reports/79: 옮기기는 도마와 그릇이 함께 보이게
                fc_ = np.mean([T.np_(o_.get_position_orientation()[0]) for o_ in objs], axis=0)
                tgt_m = np.array([0.5 * (fc_[0] + PLATE_XY[0]), 0.5 * (fc_[1] + PLATE_XY[1]), TABLE_H + 0.03])
                aim_overview(tgt_m, [0.22, -0.42, 0.55], sub=f"move_ep{seed:05d}")  # 위에서 비스듬히(팔이 덜 가리게)
                vmark("move_start")
            mrec = None
            for i_m in range(max(1, a.move_n)):  # 11-26 reports/79: 여러 조각을 차례로
                vmark(f"move{i_m}")
                mr_ = do_move(objs, recs, seed, e, args_f.veg, tag=(f"_{i_m}" if a.move_n > 1 else ""))
                if mr_ is None:
                    break
                mrec = mr_ if mrec is None else mrec
                for _ in range(int(round(0.3 / dt))):
                    run.step(arm.eef(), T.OPEN)
            vmark("move_end")
            if a.video_all and a.overview:
                for _ in range(int(round(1.0 / dt))):  # 끝 장면 1 s
                    run.step(arm.eef(), T.OPEN)
                vmark("end")
                (out / "video_marks.json").write_text(json.dumps(video_marks, indent=1))
            # 다음 에피소드 준비(기록 밖): 강체 조각 치우고 칼을 다시 쥠
            if run.food_view is not None:  # 강체를 따라가던 메쉬는 물체를 지우기 전에 숨김
                run.food_view.hide()
                run.food_view = None
            og.sim.batch_remove_objects(objs)
            for _ in range(10):
                run.step(arm.eef(), T.OPEN)
            grasp = grasp_knife_geo(T, run, knife, robot, arm, knife_geo, canonical=a.knife_grasp == "canonical")
            K_in_E = np.asarray(grasp["knife_in_eef"])
            run.K_in_E = K_in_E
            E_in_K = np.linalg.inv(K_in_E)
            if mrec is None:
                print(f"[food_robot] move seed {seed}: 옮길 조각을 못 고름(위 줄)", flush=True)
        elif a.to_rigid and objs:  # 확인만(과제 ③ 아님): 다음 에피소드 전에 강체 조각을 치움
            if run.food_view is not None:
                run.food_view.hide()
                run.food_view = None
            og.sim.batch_remove_objects(objs)
        if a.overview and run.overview_dir is not None:
            run.overview_every = 0
        run.rec, run.cmd_steps = None, None
        run.rows = []
        # 다음 에피소드: 칼을 들어 올림(쥔 채로)
        up = run.knife_pose()
        up[2, 3] += 0.08
        T.move_knife(run, K_in_E, up, T.CLOSE, speed=0.10, settle=0.2, phase="lift")
        del coupling, coupler
        g = None
        gc.collect()
    log.close()
    print(f"[food_robot] {a.episodes} 에피소드 {time.time() - t_wall:.0f} s", flush=True)
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException as _e:  # noqa: BLE001 - OmniGibson 종료 정리가 멈추거나 죽는 일이 있어(r1) 바로 끝냄
        import traceback
        traceback.print_exc()
        print(f"[food_robot] 실패: {type(_e).__name__}: {_e}", flush=True)
        sys.stdout.flush()
        os._exit(1)
