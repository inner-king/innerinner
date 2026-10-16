# SPDX-License-Identifier: Apache-2.0
"""Sim-vs-real agreement per task, as Table G.1 reports it: Pearson r and MMRV across the policies run on a task.

Paper App. J.2: both are computed per task over the policies evaluated on it, from success rates in [0, 1].
MMRV (mean maximum rank violation, from SIMPLER): for each policy i, the largest real-world gap |R_i - R_j| to
any policy j that the simulation ranks the other way round, averaged over policies:

    MMRV = 1/N * sum_i max_j |R_i - R_j| * 1[(S_i < S_j) != (R_i < R_j)]

`--check-paper` recomputes both from Table G.1's own sim and real columns and compares them with the r and MMRV
the paper prints, which pins down the conventions (ties, normalisation) before they are applied to our numbers.

With --ours <summary.json> ({task: {policy: success_rate_percent}}), each task's simulation column is ours and
the real column is the paper's; a task with fewer than three policies is reported but has no meaningful r.
"""
import argparse
import json
import math

# Table G.1 (success rate %, real / sim), per task: {policy: (real, sim)}.
TABLE_G1 = {
    "stack_dishware": {"pi0": (100, 34), "pi05": (100, 64), "gr00t_n16": (40, 0)},
    "store_marker": {"pi0": (48, 4), "pi05": (60, 20), "gr00t_n16": (32, 0)},
    "throw_away_trash": {"pi0": (20, 0), "pi05": (48, 4), "gr00t_n16": (0, 0)},
    "serve_fruits": {"pi0": (0, 4), "pi05": (72, 80), "gr00t_n16": (4, 20), "gr00t_n17": (40, 32), "dreamzero": (8, 12)},
    "cup_in_bowl": {"pi0": (88, 56), "pi05": (100, 92), "gr00t_n16": (68, 40), "gr00t_n17": (92, 92), "dreamzero": (100, 92)},
    "marker_in_cup": {"pi0": (40, 40), "pi05": (92, 88), "gr00t_n16": (28, 28), "gr00t_n17": (88, 88), "dreamzero": (88, 80)},
    "clear_table": {"pi0": (0, 12), "pi05": (40, 36), "gr00t_n16": (0, 0), "gr00t_n17": (8, 28), "dreamzero": (16, 28)},
}
PAPER_R_MMRV = {
    "stack_dishware": (0.883, 0.000), "store_marker": (0.915, 0.000), "throw_away_trash": (0.910, 0.067),
    "serve_fruits": (0.960, 0.016), "cup_in_bowl": (0.907, 0.016), "marker_in_cup": (0.995, 0.008),
    "clear_table": (0.810, 0.016),
}


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return float("nan") if sxx == 0 or syy == 0 else sxy / math.sqrt(sxx * syy)


def mmrv(sim, real):
    n = len(sim)
    worst = []
    for i in range(n):
        gaps = [abs(real[i] - real[j]) for j in range(n)
                if j != i and (sim[i] < sim[j]) != (real[i] < real[j])]
        worst.append(max(gaps, default=0.0))
    return sum(worst) / n


def agreement(pairs):
    """pairs: {policy: (real %, sim %)} -> (r, MMRV, n)."""
    real = [r / 100 for r, _ in pairs.values()]
    sim = [s / 100 for _, s in pairs.values()]
    return pearson(sim, real), mmrv(sim, real), len(pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-paper", action="store_true")
    ap.add_argument("--ours", default=None, help="JSON {task: {policy: sim success %}}")
    ap.add_argument("--json", default=None, help="write the table here")
    args = ap.parse_args()
    out = {}
    if args.check_paper:
        print(f"{'task':18s} {'n':>2s} {'r':>7s} {'paper r':>8s} {'MMRV':>7s} {'paper':>7s}")
        for task, pairs in TABLE_G1.items():
            r, m, n = agreement(pairs)
            pr, pm = PAPER_R_MMRV[task]
            print(f"{task:18s} {n:2d} {r:7.3f} {pr:8.3f} {m:7.3f} {pm:7.3f}")
            out.setdefault("paper_recomputed", {})[task] = {"r": r, "mmrv": m, "n": n, "paper_r": pr, "paper_mmrv": pm}
    if args.ours:
        ours = json.load(open(args.ours))
        print(f"\n{'task (ours sim vs paper real)':32s} {'n':>2s} {'r':>7s} {'MMRV':>7s}  policies")
        for task, sims in ours.items():
            base = task.removesuffix("_ours")
            pairs = {p: (TABLE_G1[base][p][0], s) for p, s in sims.items() if p in TABLE_G1.get(base, {})}
            r, m, n = agreement(pairs) if len(pairs) >= 2 else (float("nan"), float("nan"), len(pairs))
            note = "" if n >= 3 else "  (fewer than 3 policies: r is +-1 or undefined)"
            print(f"{task:32s} {n:2d} {r:7.3f} {m:7.3f}  {sorted(pairs)}{note}")
            out.setdefault("ours", {})[task] = {"r": r, "mmrv": m, "n": n, "pairs": pairs}
    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
