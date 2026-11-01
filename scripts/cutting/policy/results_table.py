# SPDX-License-Identifier: Apache-2.0
"""Markdown results table from eval_summary.json (one row per run, simulation failures counted separately).

Methods are compared on basic success and on the edge's lowest height (median and mean over the seeds, bootstrap
95 % intervals); the strict count is given for reference, because a quarter of the episodes flip it between
identical runs (evals_det/det_compare.md). Under a run whose episodes' places in their processes are known, a second
row leaves out each process's first episode (before 19:05 those started with the hand turned 30 deg). The last
column names the first episodes and the episodes whose hand started more than 10 deg from the run's typical start,
so neither split can hide again.
"""
import json
import sys

rows = json.load(open(sys.argv[1]))
kinds_ko = {"missed": "빗나감", "incomplete": "덜 잘림", "force": "힘 초과", "knife": "칼 놓침", "board_dig": "도마 파고듦",
            "board_push": "도마 밀림", "sim": "시뮬레이션 실패"}


def edge(r):
    e = r.get("edge")
    if not e:
        return "-"
    return (f"{e['median']:+.2f} [{e['median_boot95'][0]:+.2f}, {e['median_boot95'][1]:+.2f}] · "
            f"{e['mean']:+.2f} [{e['mean_boot95'][0]:+.2f}, {e['mean_boot95'][1]:+.2f}] ({e['ref']})")


def bottom(r):
    b = r.get("bottom")
    if not b:
        return "-"
    return (f"{b['p10']:+.2f} [{b['p10_boot95'][0]:+.2f}, {b['p10_boot95'][1]:+.2f}] · "
            f"{b['sd']:.2f} [{b['sd_boot95'][0]:.2f}, {b['sd_boot95'][1]:.2f}] ({b['episodes']})")


def line(label, r, last):
    n = r["n"]
    b, s = r["basic"], r["strict"]
    bw, sw = r["basic_wilson95"], r["strict_wilson95"]
    k = r["kinds"]
    ks = ", ".join(f"{kinds_ko.get(a, a)} {v}" for a, v in sorted(k.items(), key=lambda x: -x[1])) or "-"
    print(f"| {label} | {n} | {b}/{n} = {100 * b / n:.0f}% [{100 * bw[0]:.0f}, {100 * bw[1]:.0f}] | {edge(r)} | {bottom(r)} | "
          f"{s}/{n} = {100 * s / n:.0f}% [{100 * sw[0]:.0f}, {100 * sw[1]:.0f}] | {k.get('sim', 0)} | {ks} | {last} |")


tofu_ref = [(label, r) for label, r in rows.items() if (r.get("edge") or {}).get("ref") == "두부 바닥"]
if tofu_ref:  # the LiDAR scene: its strict count leaves out the dig rule, which must not read as "no depth problem"
    med = [r["edge"]["median"] for label, r in tofu_ref if "전문가" not in label and "눈 감은" not in label]
    span_txt = (f"{min(med):+.2f} ~ {max(med):+.2f} mm" if med else "-")
    print(f"> (나) 주의: 이 표의 엄격 성공에는 도마 파고듦 판정이 빠져 있습니다((나)의 도마 윗면 측정이 틀려서). "
          f"날 최저 높이(두부 바닥면 기준)의 정책 중앙값은 {span_txt}로 두부 바닥 아래 약 3 mm까지 내려갑니다"
          f"(스크립트 전문가 +0.94 mm). 두부가 도마 위에 놓여 있으므로 칼이 도마를 약 3 mm 파고드는 것으로 보입니다"
          f"[추정: 도마 윗면을 직접 재지 못함]. 그래서 (나)의 엄격 성공은 깊이에 문제가 없다는 뜻이 아닙니다. "
          f"도마 윗면 측정이 고쳐지면 같은 파고듦 기준으로 다시 셉니다(19:25 이후 에피소드는 날의 x, y, z를 모두 기록).\n")
print("| 방법 | 에피소드 | 기본 성공 [95% 구간] | 날 최저 높이 mm: 중앙값 [95%] · 평균 [95%] (기준) "
      "| 바닥 구간(기준 위 5 mm 아래) 날 높이 mm: 10번째 백분위 · 표준편차, 에피소드별 값의 중앙값 [95%] (에피소드 수) "
      "| 엄격 성공(참고) [95% 구간] | 시뮬레이션 실패 | 실패 유형(겹침 있음) | 프로세스 첫 에피소드 / 손이 10° 넘게 돌아간 시작 |")
print("|---|---|---|---|---|---|---|---|---|")
def span(vals, fmt):
    return fmt(min(vals)) if min(vals) == max(vals) else f"{fmt(min(vals))} ~ {fmt(max(vals))}"


groups = {}
for label, r in rows.items():
    if "first_seeds" not in r:  # a summary written before 19:05
        line(label, r, "-")
        continue
    firsts = ", ".join(map(str, r["first_seeds"])) if r["order_known"] else "모름"
    turned = ", ".join(map(str, r["turned_start_seeds"])) or "없음"
    line(label + " — 모든 에피소드", r, f"{firsts} / {turned}")
    if "excl_first" in r:
        line(label + " — 첫 에피소드 제외", r["excl_first"], "제외함")
    rep = r.get("repeat")
    if rep:  # the same evaluation run again: after its last run, the range the runs spanned
        groups.setdefault(rep["group"], []).append(r)
        if rep["index"] == rep["of"]:
            g = groups[rep["group"]]
            med = [x["edge"]["median"] for x in g if x.get("edge")]
            print(f"| {rep['group']} — 반복 범위 ({len(g)}회) | {g[0]['n']} | {span([x['basic'] for x in g], str)} "
                  f"| 중앙값 {span(med, lambda v: f'{v:+.2f}') if med else '-'} | | {span([x['strict'] for x in g], str)} "
                  f"| | | |")
