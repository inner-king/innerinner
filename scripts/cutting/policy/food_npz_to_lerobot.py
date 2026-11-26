"""food_robot_episodes.py 의 에피소드(ep_*.npz + .json, 팀원 꼴)를 팀원 변환기 npz_to_lerobot.py 로 LeRobot 데이터셋으로 (2026-11-24, reports/66).

팀원 변환기는 과제 문장이 두부로 박혀 있다(TASK = "cut the tofu on the board into two pieces"). 이 감싸개는 그 값만 우리 과제 문장으로
바꾸고 나머지(관측·행동 열, 성공 에피소드만, 영상 인코딩)는 팀원 것 그대로 쓴다.

  python scripts/cutting/food/food_npz_to_lerobot.py --src <에피소드 폴더> --root <데이터셋 폴더> [팀원 옵션…]
  과제 문장: --task 를 주거나, 없으면 첫 ep_*.json 의 "task".
팀원 변환기 위치: --team <이 저장소>(기본).

11-26(reports/81) --gripper_action: 행동 9 칸(손 자리 3 + 회전 6) 뒤에 10 번째 칸 'gripper'(다음 틱 집게 명령, 팀원 T.OPEN 1 / T.CLOSE 0)를 붙인다.
  놓기·옮기기는 집게를 열고 닫는데 팀원 꼴(두부 썰기 — 칼을 계속 쥠)에는 집게 명령이 없어서다. 값: npz 의 grip_cmd(11-26 뒤 에피소드)가 있으면 그것,
  없으면 단계 이름으로(옮기기·놓기: approach·descend·release·retreat = 열기, 나머지 = 닫기 / 칼을 쥔 과제(썰기·떼기·떼어 놓기) = 늘 닫기).
  팀원 --action delta · --drop-near-constant · --action-height log-support 와는 같이 못 씀. 기본은 꺼짐(팀원 꼴 9 칸 그대로).
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

OPEN_PHASES = {"approach", "descend", "release", "retreat"}
GRASP_MODES = ("expert_move", "expert_place")


def grip_column(d, info):
    """에피소드 npz 내용(dict)과 json → 틱마다 집게 명령(1 열기 / 0 닫기)."""
    g = d.get("grip_cmd")
    if g is not None and len(g) == len(d["cmd"]) and np.isfinite(g).all():
        return np.asarray(g, np.float32)
    ph = [str(x) for x in d.get("tick_phase", [""] * len(d["cmd"]))]
    if str(info.get("mode", "")) in GRASP_MODES:
        return np.array([1.0 if p in OPEN_PHASES else 0.0 for p in ph], np.float32)
    return np.zeros(len(d["cmd"]), np.float32)  # 칼을 쥔 과제


class _NpWithGrip:
    """팀원 변환기 안의 np 를 감싸 np.load(ep_*.npz) 가 cmd 에 집게 칸을 붙인 dict 를 돌려주게 함(나머지는 numpy 그대로)."""

    def __getattr__(self, n):
        return getattr(np, n)

    def load(self, p, *args, **kw):
        z = np.load(p, *args, **kw)
        if not str(p).endswith(".npz"):
            return z
        d = {k: z[k] for k in z.files}
        js = Path(str(p)).with_suffix(".json")
        info = json.loads(js.read_text()) if js.exists() else {}
        d["cmd"] = np.concatenate([d["cmd"], grip_column(d, info)[:, None]], axis=1).astype(np.float32)
        return d


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--task", default=None)
    ap.add_argument("--team", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--gripper_action", action="store_true")
    a, rest = ap.parse_known_args()
    conv = Path(a.team) / "scripts" / "cutting" / "policy" / "npz_to_lerobot.py"
    spec = importlib.util.spec_from_file_location("npz_to_lerobot", conv)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    task = a.task
    if task is None:
        srcs = [Path(rest[i + 1]) for i, v in enumerate(rest) if v == "--src" and i + 1 < len(rest)]
        for d in srcs:
            js = sorted(d.glob("ep_*.json"))
            if js:
                task = json.loads(js[0].read_text()).get("task")
                break
    if task:
        mod.TASK = task
    if a.gripper_action:
        bad = [o for o in ("--drop-near-constant", "log-support", "delta") if o in rest]
        if bad:
            raise SystemExit(f"--gripper_action 은 {bad} 와 같이 못 씀")
        mod.ACTION = list(mod.ACTION) + ["gripper"]
        mod.np = _NpWithGrip()
    print(f"[food_npz_to_lerobot] 과제 문장: {mod.TASK!r}, 행동 {len(mod.ACTION)} 칸 (팀원 변환기 {conv})", flush=True)
    sys.argv = [str(conv)] + rest
    mod.main()


if __name__ == "__main__":
    main()
