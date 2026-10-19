# SPDX-License-Identifier: Apache-2.0
"""Sim-vs-real agreement per task, as Table G.1 reports it: Pearson r and MMRV across the policies run on a task.

Paper App. J.2: both are computed per task over the policies evaluated on it, from success rates in [0, 1].
MMRV (mean maximum rank violation, from SIMPLER): for each policy i, the largest real-world gap |R_i - R_j| to
any policy j that the simulation ranks the other way round, averaged over policies:

    MMRV = 1/N * sum_i max_j |R_i - R_j| * 1[(S_i < S_j) != (R_i < R_j)]

`--check-paper` recomputes both from Table G.1's own sim and real columns and compares them with the r and MMRV
the paper prints, which pins down the conventions (ties, normalisation) before they are applied to our numbers.

Our runs are read with --results ROOT:TAG (eval_results.json under ROOT/<task>/s15_eval_<policy>_<tag>/, as the
eval stage writes them; repeatable) and --summary FILE (the list run_table_g1.sh's summarize.py writes; repeatable).
Each task's simulation column is then ours and the real column the paper's. A task with fewer than three policies
is listed but has no meaningful r. Task names containing _ours (and variants such as _ours_near) are our reconstructed scenes, compared with
the paper's row of the same task.

`--draws N` adds a 95% interval to each of our r values: every policy's success rate is drawn N times from its
posterior under a uniform prior, Beta(k + 1, n - k + 1), and r is recomputed against the paper's real column. Three
policies at 25 episodes each leave r very loose. A bootstrap that resamples the observed rate would treat a 25/25
as certain and, with two policies at 100%, report no spread at all.
"""
import argparse
import glob
import json
import math
import os
import random

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


def table_row(task):
    """The Table G.1 row a run's task or scene name belongs to: the longest row name it starts with
    (marker_in_cup_ours_mass, marker_in_cup_kp4 -> marker_in_cup)."""
    return max((k for k in TABLE_G1 if task.startswith(k)), key=len, default=task.split("_ours")[0])


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


# Our policy names (run directories) -> Table G.1 columns.
POLICY_COLUMN = {"pi0": "pi0", "pi05": "pi05", "gr00t": "gr00t_n16", "gr00t17": "gr00t_n17"}


def collect_results(root_tag):
    """{(task, policy): (successes, episodes)} from the newest eval_results.json per task and policy."""
    root, _, tag = root_tag.partition(":")
    tag = tag or "main"
    found = {}
    for path in glob.glob(os.path.join(root, "*", f"s15_eval_*_{tag}", "*", "*", "results_*", "eval_results.json")):
        task = path[len(root):].lstrip(os.sep).split(os.sep)[0]
        policy = path.split(os.sep)[-5][len("s15_eval_"):-len(f"_{tag}")]
        if (task, policy) not in found or os.path.getmtime(path) > os.path.getmtime(found[(task, policy)]):
            found[(task, policy)] = path
    out = {}
    for key, path in found.items():
        eps = json.load(open(path))["episodes"]
        out[key] = (sum(bool(e["success"]) for e in eps), len(eps))
    return out


def collect_summary(path):
    return {(r["task"], r["policy"]): (r["successes"], r["episodes"]) for r in json.load(open(path))}


def agreement(pairs):
    """pairs: {policy: (real %, sim %)} -> (r, MMRV, n)."""
    real = [r / 100 for r, _ in pairs.values()]
    sim = [s / 100 for _, s in pairs.values()]
    return pearson(sim, real), mmrv(sim, real), len(pairs)


def r_interval(counts, real, draws, seed=0):
    """2.5th and 97.5th percentiles of r over success rates drawn from Beta(k + 1, n - k + 1).

    Args:
        counts (list): (successes, episodes) per policy
        real (list): the paper's real-world success rate (%) per policy, in the same order
        draws (int): number of draws
        seed (int): seed of the draws, so a run reproduces its interval
    """
    rng = random.Random(seed)
    rs = sorted(r for r in (pearson([rng.betavariate(k + 1, n - k + 1) for k, n in counts], [x / 100 for x in real])
                            for _ in range(draws)) if not math.isnan(r))
    return rs[int(0.025 * (len(rs) - 1))], rs[int(0.975 * (len(rs) - 1))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-paper", action="store_true")
    ap.add_argument("--results", action="append", default=[], help="ROOT:TAG of eval-stage outputs")
    ap.add_argument("--summary", action="append", default=[], help="summarize.py --json output")
    ap.add_argument("--json", default=None, help="write the table here")
    ap.add_argument("--draws", type=int, default=0, help="add a 95%% interval for r from this many posterior draws")
    args = ap.parse_args()
    out = {}
    if args.check_paper:
        print(f"{'task':18s} {'n':>2s} {'r':>7s} {'paper r':>8s} {'MMRV':>7s} {'paper':>7s}")
        for task, pairs in TABLE_G1.items():
            r, m, n = agreement(pairs)
            pr, pm = PAPER_R_MMRV[task]
            print(f"{task:18s} {n:2d} {r:7.3f} {pr:8.3f} {m:7.3f} {pm:7.3f}")
            out.setdefault("paper_recomputed", {})[task] = {"r": r, "mmrv": m, "n": n, "paper_r": pr, "paper_mmrv": pm}
    runs = {}
    for rt in args.results:
        runs.update(collect_results(rt))
    for sm in args.summary:
        runs.update(collect_summary(sm))
    if runs:
        print(f"\n| task | policy | ours (sim) | paper sim | paper real |\n|---|---|---|---|---|")
        tasks, counts = {}, {}
        for (task, policy), (k, n) in sorted(runs.items()):
            col = POLICY_COLUMN.get(policy, policy)
            real, sim = TABLE_G1.get(table_row(task), {}).get(col, (None, None))
            print(f"| {task} | {col} | {k}/{n} = {100 * k / n:.0f}% | {sim} | {real} |")
            out.setdefault("runs", []).append({"task": task, "policy": col, "successes": k, "episodes": n,
                                               "paper_sim": sim, "paper_real": real})
            if real is not None:
                tasks.setdefault(task, {})[col] = (real, 100 * k / n)
                counts.setdefault(task, {})[col] = (k, n)
        interval = " r 95% interval |" if args.draws else ""
        print(f"\n| task (our sim vs paper real) | policies | r |{interval} MMRV | paper r | paper MMRV |\n"
              f"|---|---|---|{'---|' if args.draws else ''}---|---|---|")
        for task, pairs in tasks.items():
            r, m, n = agreement(pairs) if len(pairs) >= 2 else (float("nan"), float("nan"), len(pairs))
            pr, pm = PAPER_R_MMRV[table_row(task)]
            note = "" if n >= 3 else " (r is +-1 or undefined with fewer than 3)"
            row = {"r": r, "mmrv": m, "n": n, "paper_r": pr, "paper_mmrv": pm}
            cell = ""
            if args.draws and n >= 3:
                row["r_interval_95"] = r_interval(list(counts[task].values()), [real for real, _ in pairs.values()],
                                                  args.draws)
                cell = f" {row['r_interval_95'][0]:+.2f}..{row['r_interval_95'][1]:+.2f} |"
            elif args.draws:
                cell = " |"
            print(f"| {task} | {', '.join(sorted(pairs))}{note} | {r:.3f} |{cell} {m:.3f} | {pr:.3f} | {pm:.3f} |")
            out.setdefault("agreement", {})[task] = row
    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
