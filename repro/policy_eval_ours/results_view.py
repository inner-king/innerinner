# SPDX-License-Identifier: Apache-2.0
"""A shallow summary of every Table G.1 policy-evaluation run, next to the deep raw outputs it points into.

The eval stage writes each run 5 folders deep (<scene>/s15_eval_<policy>_<tag>/<family>/<checkpoint>/results_<time>/),
under a root per experiment and machine, so the runs are hard to browse on the Drive. This reads them all and
writes, into one folder (default results/results_view/policy_eval/):

    runs.csv / runs.md     one row per run: task, scene, what was changed, policy, successes, the paper's
                           simulation and real-world rates, the README chapter, and the raw folder's Drive path
    agreement.csv / .md    per task and scene variant with all three policies: Pearson r (with the 95% interval
                           of correlate.py --draws), MMRV, and the paper's values
    videos/                one success and one failure per run, renamed <scene>__<policy>__<tag>__epNNN_<result>.mp4

The raw folders are only read. Smoke runs (tags containing "smoke" or "check") are left out, and so are the copies
of Colab results kept under ~/simfoundry_logs/policy_eval_<experiment>/, which results/policy_eval_ours/ also holds.

    python repro/policy_eval_ours/results_view.py
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from correlate import TABLE_G1, PAPER_R_MMRV, POLICY_COLUMN, mmrv, pearson, r_interval  # noqa: E402

# Local root -> where repro/drive_sync.sh puts it under the Drive folder innerinner/.
ROOTS = [(os.path.join(REPO, "results", "policy_eval_ours"), "results/policy_eval_ours"),
         (os.path.expanduser("~/simfoundry_logs/policy_eval"), "results/logs/policy_eval")]
# Runs whose raw folders did not survive, known by their success counts only (records with scene, policy, tag,
# successes, episodes, source).
COUNT_FILES = [os.path.join(REPO, "results", "diagnostics", "policy_eval_ours", "serve_fruits",
                            "authplate_openpi_counts.json")]

# (scene, tag) -> (README chapter, what the run changed). Runs not listed still appear, undescribed.
AUTHORS = "저자 장면"
DESCRIBE = {
    ("marker_in_cup", "main"): ("18, 19", AUTHORS),
    ("serve_fruits", "main"): ("18, 19", AUTHORS),
    ("stack_dishware", "main"): ("18", AUTHORS + " (미세조정 과제에 공개 체크포인트)"),
    ("store_marker", "main"): ("18", AUTHORS + " (미세조정 과제에 공개 체크포인트)"),
    ("throw_away_trash", "main"): ("18", AUTHORS + " (미세조정 과제에 공개 체크포인트)"),
    ("marker_in_cup", "calcam"): ("18", AUTHORS + ", 카메라를 실제 보정값으로"),
    ("marker_in_cup", "mug300g"): ("18", AUTHORS + ", 머그 질량 300 g"),
    ("marker_in_cup", "prompt_repo"): ("18", AUTHORS + ", 저장소 설정의 지시문"),
    ("marker_in_cup", "h15"): ("18.4", AUTHORS + ", 실행 지평 15"),
    ("marker_in_cup_tight", "main"): ("18.4", AUTHORS + ", 마커 격자 ±5 cm"),
    ("marker_in_cup_tight", "h15"): ("18.4", AUTHORS + ", 마커 격자 ±5 cm + 지평 15"),
    ("marker_in_cup", "linear"): ("20.2", AUTHORS + ", 톤매퍼 linear"),
    ("serve_fruits", "linear"): ("20.2", AUTHORS + ", 톤매퍼 linear"),
    ("marker_in_cup", "norobot"): ("23", AUTHORS + ", 배경 스캔의 로봇을 가림"),
    ("marker_in_cup", "physical"): ("26.4", AUTHORS + ", physical 잡기"),
    ("marker_in_cup_kp4", "main"): ("26", AUTHORS + ", 팔 관절 stiffness ×4 (추적 지연 0.2 → 0.04 s)"),
    ("marker_in_cup_limits", "main"): ("28", AUTHORS + ", 손가락 관절의 속도·힘 제한 적용"),
    ("marker_in_cup_limits", "rep2"): ("28", AUTHORS + ", 손가락 관절의 속도·힘 제한 적용, 반복 2"),
    ("cup_in_bowl_ours", "ours"): ("19", "우리 복원"),
    ("cup_in_bowl_ours", "main"): ("19", "우리 복원"),
    ("cup_in_bowl_ours_near", "near"): ("19", "우리 복원, 가까운 배치"),
    ("cup_in_bowl_ours_near", "main"): ("19", "우리 복원, 가까운 배치"),
    ("marker_in_cup_ours", "ours"): ("19", "우리 복원"),
    ("marker_in_cup_ours", "main"): ("19", "우리 복원"),
    ("cup_in_bowl_ours_mass", "mass"): ("21", "우리 복원, 질량을 실물에 가깝게"),
    ("cup_in_bowl_ours_cfg", "main"): ("25", "우리 복원, 질량 수정, 저장소 태스크 설정의 배치"),
    ("cup_in_bowl_ours_cfg", "physical"): ("27", "우리 복원, 질량 수정, 저장소 태스크 설정의 배치, physical 잡기"),
    ("cup_in_bowl_ours_real", "main"): ("30", "우리 복원, 11단계 새 질량, 손가락 제한, physical 잡기"),
    ("cup_in_bowl_ours_real", "rep2"): ("30", "우리 복원, 11단계 새 질량, 손가락 제한, physical 잡기, 반복 2"),
    ("cup_in_bowl_ours_real", "rep3"): ("30", "우리 복원, 11단계 새 질량, 손가락 제한, physical 잡기, 반복 3"),
    ("cup_in_bowl_ours_real", "rep4"): ("30", "우리 복원, 11단계 새 질량, 손가락 제한, physical 잡기, 반복 4"),
    ("cup_in_bowl_ours_real_authbowl", "main"): ("30.7", "우리 복원, 모든 수정, 그릇만 저자 YAM 장면의 것"),
    ("cup_in_bowl_ours_real_authbowl", "rep2"): ("30.7", "우리 복원, 모든 수정, 그릇만 저자 YAM 장면의 것, 반복 2"),
    ("marker_in_cup_ours_real", "assisted"): ("30.6", "우리 복원, 모든 수정이지만 assisted 잡기"),
    ("marker_in_cup_ours_real", "assisted_rep2"): ("30.6", "우리 복원, 모든 수정이지만 assisted 잡기, 반복 2"),
    ("serve_fruits_ours_real", "main"): ("30", "우리 복원, 11단계 새 질량, 손가락 제한, physical 잡기"),
    ("serve_fruits_ours_real", "rep2"): ("30", "우리 복원, 11단계 새 질량, 손가락 제한, physical 잡기, 반복 2"),
    ("marker_in_cup_ours_mass", "mass"): ("21", "우리 복원, 질량을 실물에 가깝게"),
    ("marker_in_cup_ours_real", "main"): ("30", "우리 복원, 11단계 새 질량, 손가락 제한, physical 잡기"),
    ("marker_in_cup_ours_real", "rep2"): ("30", "우리 복원, 11단계 새 질량, 손가락 제한, physical 잡기, 반복 2"),
    ("serve_fruits_ours", "main"): ("22", "우리 복원"),
    ("serve_fruits_ours_authmass", "main"): ("22", "우리 복원, 저자 물체의 질량"),
    ("serve_fruits_ours_authplate", "main"): ("22", "우리 복원, 주황 접시만 저자 것"),
}
for rep in (2, 3, 4):
    DESCRIBE[("marker_in_cup", f"rep{rep}")] = ("18.4", AUTHORS + f", 기준 반복 {rep}")
    DESCRIBE[("marker_in_cup_kp4", f"rep{rep}")] = ("26.3", AUTHORS + f", 팔 stiffness ×4, 반복 {rep}")
    DESCRIBE[("marker_in_cup_tight", f"h15_rep{rep}")] = ("18.4", AUTHORS + f", 마커 격자 ±5 cm + 지평 15, 반복 {rep}")

# Three-policy sets whose r and MMRV the README reports: (task, scene variant label, {policy: (scene, tag)}).
SETS = [
    ("marker_in_cup", AUTHORS, {"pi0": ("marker_in_cup", "main"), "pi05": ("marker_in_cup", "main"),
                                "gr00t": ("marker_in_cup", "main")}),
    ("marker_in_cup", AUTHORS + ", 로봇 가림", {"pi0": ("marker_in_cup", "norobot"), "pi05": ("marker_in_cup", "norobot"),
                                              "gr00t": ("marker_in_cup", "norobot")}),
    ("marker_in_cup", AUTHORS + ", 팔 stiffness ×4", {"pi0": ("marker_in_cup_kp4", "main"),
                                                    "pi05": ("marker_in_cup_kp4", "main"),
                                                    "gr00t": ("marker_in_cup_kp4", "main")}),
    ("marker_in_cup", AUTHORS + ", 손가락 제한", {"pi0": ("marker_in_cup_limits", "main"),
                                              "pi05": ("marker_in_cup_limits", "main"),
                                              "gr00t": ("marker_in_cup_limits", "main")}),
    ("marker_in_cup", AUTHORS + ", physical 잡기", {"pi0": ("marker_in_cup", "physical"),
                                                 "pi05": ("marker_in_cup", "physical"),
                                                 "gr00t": ("marker_in_cup", "physical")}),
    ("serve_fruits", AUTHORS, {"pi0": ("serve_fruits", "main"), "pi05": ("serve_fruits", "main"),
                               "gr00t": ("serve_fruits", "main")}),
    ("cup_in_bowl", "우리 복원", {"pi0": ("cup_in_bowl_ours", "ours"), "pi05": ("cup_in_bowl_ours", "ours"),
                                "gr00t": ("cup_in_bowl_ours", "main")}),
    ("cup_in_bowl", "우리 복원, 가까운 배치", {"pi0": ("cup_in_bowl_ours_near", "near"),
                                         "pi05": ("cup_in_bowl_ours_near", "near"),
                                         "gr00t": ("cup_in_bowl_ours_near", "main")}),
    ("cup_in_bowl", "우리 복원, 질량 수정, 저장소 설정 배치", {"pi0": ("cup_in_bowl_ours_cfg", "main"),
                                                     "pi05": ("cup_in_bowl_ours_cfg", "main"),
                                                     "gr00t": ("cup_in_bowl_ours_cfg", "main")}),
    ("cup_in_bowl", "우리 복원, 질량 수정, 저장소 설정 배치, physical 잡기",
     {"pi0": ("cup_in_bowl_ours_cfg", "physical"), "pi05": ("cup_in_bowl_ours_cfg", "physical"),
      "gr00t": ("cup_in_bowl_ours_cfg", "physical")}),
    ("cup_in_bowl", "우리 복원, 11단계 새 질량, 손가락 제한, physical 잡기",
     {"pi0": ("cup_in_bowl_ours_real", "main"), "pi05": ("cup_in_bowl_ours_real", "main"),
      "gr00t": ("cup_in_bowl_ours_real", "main")}),
    ("cup_in_bowl", "우리 복원, 질량 수정", {"pi0": ("cup_in_bowl_ours_mass", "mass"),
                                       "pi05": ("cup_in_bowl_ours_mass", "mass"),
                                       "gr00t": ("cup_in_bowl_ours_mass", "mass")}),
    ("marker_in_cup", "우리 복원", {"pi0": ("marker_in_cup_ours", "ours"), "pi05": ("marker_in_cup_ours", "ours"),
                                  "gr00t": ("marker_in_cup_ours", "main")}),
    ("marker_in_cup", "우리 복원, 질량 수정", {"pi0": ("marker_in_cup_ours_mass", "mass"),
                                         "pi05": ("marker_in_cup_ours_mass", "mass"),
                                         "gr00t": ("marker_in_cup_ours_mass", "mass")}),
    ("marker_in_cup", "우리 복원, 11단계 새 질량, 손가락 제한, physical 잡기",
     {"pi0": ("marker_in_cup_ours_real", "main"), "pi05": ("marker_in_cup_ours_real", "main"),
      "gr00t": ("marker_in_cup_ours_real", "main")}),
    ("serve_fruits", "우리 복원", {"pi0": ("serve_fruits_ours", "main"), "pi05": ("serve_fruits_ours", "main"),
                                 "gr00t": ("serve_fruits_ours", "main")}),
    ("serve_fruits", "우리 복원, 주황 접시만 저자 것", {"pi0": ("serve_fruits_ours_authplate", "main"),
                                                "pi05": ("serve_fruits_ours_authplate", "main"),
                                                "gr00t": ("serve_fruits_ours_authplate", "main")}),
]


def task_of(scene):
    """The Table G.1 row a scene belongs to: marker_in_cup_ours_mass -> marker_in_cup."""
    return max((k for k in TABLE_G1 if scene.startswith(k)), key=len, default=scene.split("_ours")[0])


def find_runs():
    runs = []
    for root, drive in ROOTS:
        for path in sorted(glob.glob(os.path.join(root, "**", "eval_results.json"), recursive=True)):
            parts = os.path.relpath(path, root).split(os.sep)
            run_dir = next((p for p in parts if p.startswith("s15_eval_")), None)
            if run_dir is None:
                continue
            scene = parts[parts.index(run_dir) - 1]
            rest = run_dir[len("s15_eval_"):]
            policy = next((p for p in ("pi05", "pi0", "gr00t") if rest.startswith(p + "_")), None)
            tag = rest[len(policy) + 1:] if policy else rest
            if policy is None or "smoke" in tag or "check" in tag:
                continue
            episodes = json.load(open(path))["episodes"]
            runs.append({"scene": scene, "task": task_of(scene), "policy": policy, "tag": tag,
                         "successes": sum(bool(e["success"]) for e in episodes), "episodes": len(episodes),
                         "results": os.path.basename(os.path.dirname(path)), "dir": os.path.dirname(path),
                         "drive": os.path.join(drive, os.path.relpath(os.path.dirname(path), root))})
    # The newest results folder per (scene, policy, tag), as correlate.py reads them.
    newest = {}
    for r in runs:
        key = (r["scene"], r["policy"], r["tag"])
        if key not in newest or r["results"] > newest[key]["results"]:
            newest[key] = r
    for path in COUNT_FILES:
        for c in (json.load(open(path)) if os.path.exists(path) else []):
            key = (c["scene"], c["policy"], c["tag"])
            if key not in newest:
                newest[key] = {"scene": c["scene"], "task": task_of(c["scene"]), "policy": c["policy"], "tag": c["tag"],
                               "successes": c["successes"], "episodes": c["episodes"], "results": "", "dir": None,
                               "drive": f"(원본 없음, 성공 횟수만: {os.path.relpath(path, REPO)})"}
    return sorted(newest.values(), key=lambda r: (r["task"], r["scene"], r["tag"], r["policy"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "results_view", "policy_eval"))
    ap.add_argument("--draws", type=int, default=20000)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "videos"), exist_ok=True)
    runs = find_runs()

    rows = []
    for r in runs:
        chapter, what = DESCRIBE.get((r["scene"], r["tag"]), ("", ""))
        real, sim = TABLE_G1.get(r["task"], {}).get(POLICY_COLUMN[r["policy"]], (None, None))
        videos = sorted(glob.glob(os.path.join(r["dir"], "videos", "episode_*.mp4"))) if r["dir"] else []
        picks = [next((v for v in videos if v.endswith(f"_{kind}.mp4")), None) for kind in ("success", "fail")]
        copied = []
        for v in filter(None, picks):
            # episode_003_success.mp4 -> <scene>__<policy>__<tag>__ep003_success.mp4
            name = f"{r['scene']}__{r['policy']}__{r['tag']}__ep{os.path.basename(v)[len('episode_'):]}"
            shutil.copy2(v, os.path.join(args.out, "videos", name))
            copied.append(name)
        rows.append({"task": r["task"], "scene": r["scene"], "variant": what, "policy": r["policy"], "tag": r["tag"],
                     "successes": r["successes"], "episodes": r["episodes"],
                     "rate_pct": round(100 * r["successes"] / r["episodes"]),
                     "paper_sim_pct": sim, "paper_real_pct": real, "chapter": chapter,
                     "videos": " ".join(copied), "raw_on_drive": r["drive"]})
    with open(os.path.join(args.out, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lookup = {(r["scene"], r["policy"], r["tag"]): r for r in runs}
    sets = []
    for task, label, members in SETS:
        got = {p: lookup.get((scene, p, tag)) for p, (scene, tag) in members.items()}
        if not all(got.values()):
            continue
        real = [TABLE_G1[task][POLICY_COLUMN[p]][0] for p in got]
        sim = [100 * g["successes"] / g["episodes"] for g in got.values()]
        lo, hi = r_interval([(g["successes"], g["episodes"]) for g in got.values()], real, args.draws)
        sets.append({"task": task, "scene": label,
                     "pi0": f"{got['pi0']['successes']}/{got['pi0']['episodes']}",
                     "pi05": f"{got['pi05']['successes']}/{got['pi05']['episodes']}",
                     "gr00t": f"{got['gr00t']['successes']}/{got['gr00t']['episodes']}",
                     "r": round(pearson([s / 100 for s in sim], [x / 100 for x in real]), 3),
                     "r_95": f"{lo:+.2f}..{hi:+.2f}",
                     "mmrv": round(mmrv([s / 100 for s in sim], [x / 100 for x in real]), 3),
                     "paper_r": PAPER_R_MMRV[task][0], "paper_mmrv": PAPER_R_MMRV[task][1]})
    with open(os.path.join(args.out, "agreement.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sets[0]))
        w.writeheader()
        w.writerows(sets)

    with open(os.path.join(args.out, "runs.md"), "w") as f:
        f.write("| 과제 | 장면 | 무엇을 바꿨나 | 정책 | 성공 | 논문 시뮬 | 논문 실제 | 장 | 원본 (Drive innerinner/) |\n"
                "|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['task']} | {r['scene']} | {r['variant']} | {r['policy']} | {r['successes']}/{r['episodes']} "
                    f"= {r['rate_pct']}% | {r['paper_sim_pct'] if r['paper_sim_pct'] is not None else ''} "
                    f"| {r['paper_real_pct'] if r['paper_real_pct'] is not None else ''} | {r['chapter']} "
                    f"| `{r['raw_on_drive']}` |\n")
    with open(os.path.join(args.out, "agreement.md"), "w") as f:
        f.write("| 과제 | 장면 | π0 | π0.5 | GR00T N1.6 | r | r 95% 구간 | MMRV | 논문 r | 논문 MMRV |\n"
                "|---|---|---|---|---|---|---|---|---|---|\n")
        for s in sets:
            f.write(f"| {s['task']} | {s['scene']} | {s['pi0']} | {s['pi05']} | {s['gr00t']} | {s['r']} | {s['r_95']} "
                    f"| {s['mmrv']} | {s['paper_r']} | {s['paper_mmrv']} |\n")
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write("# Table G.1 정책 평가 요약\n\n"
                "`repro/policy_eval_ours/results_view.py`가 만든 폴더입니다. 원본은 깊은 폴더에 그대로 있고, 여기에는 "
                "그 요약과 대표 영상만 있습니다. 다시 만들려면 저장소에서 위 스크립트를 실행하면 됩니다.\n\n"
                "- `agreement.md` / `.csv`: 과제와 장면마다 정책 세 개(π0, π0.5, GR00T N1.6)의 성공 횟수, 논문 실제 "
                "성공률과의 Pearson r(95% 구간은 성공률을 Beta(k+1, n−k+1)에서 2만 번 뽑아 다시 계산), MMRV, 논문 값\n"
                "- `runs.md` / `.csv`: 실행 하나에 한 줄. 무엇을 바꿨는지, 성공 횟수, 논문의 시뮬·실제 성공률, "
                "자세한 설명이 있는 `repro/README.md`의 장, 원본 폴더의 Drive 경로(`innerinner/` 기준)\n"
                "- `videos/`: 실행마다 성공 하나와 실패 하나. 이름은 `<장면>__<정책>__<태그>__ep<번호>_<결과>.mp4`이고, "
                "화면은 왼쪽부터 외부 카메라 두 대와 손목 카메라입니다\n\n"
                "연기 시험(태그에 smoke, check)은 뺐습니다. 같은 실행의 결과 폴더가 여럿이면 가장 최근 것을 씁니다"
                "(correlate.py와 같음).\n")
    print(f"[view] {args.out}: {len(rows)} runs, {len(sets)} three-policy sets, "
          f"{len(os.listdir(os.path.join(args.out, 'videos')))} videos")


if __name__ == "__main__":
    main()
