# SPDX-License-Identifier: Apache-2.0
"""Pool repeated runs of one setting and compare settings (repro/README.md, 18.4).

Each --arm names a setting and the runs that played it (results_* directories or eval_results.json files),
for instance one start file evaluated several times. Prints, per arm, each run's successes, the pooled rate
with a Wilson 95% interval, the median length of the successful episodes, and whether the runs agree with one
another (a permutation test of "every run has the same success probability", whose statistic is the
chi-square over runs; pooling assumes it). Then Fisher's exact two-sided p between every pair of arms, and of
each arm against --paper (successes/episodes; the paper ran 25 episodes per cell, App. J.1).

    python repro/policy_eval/pool_runs.py --paper 22/25 \
        --arm baseline out/marker_in_cup/s15_eval_pi05_{main,rep2,rep3,rep4}/*/*/results_* \
        --arm combined out/marker_in_cup_tight/s15_eval_pi05_{h15,h15_rep2,h15_rep3,h15_rep4}/*/*/results_*
"""
import argparse
import itertools
import json
import os

import numpy as np

from summarize import fisher_two_sided, wilson


def load_episodes(run):
    path = run if run.endswith(".json") else os.path.join(run, "eval_results.json")
    return json.load(open(path))["episodes"]


def rate(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} ({lo:.0%}-{hi:.0%})"


def chi2_over_runs(succ, run_of, n_runs):
    p = succ.mean()
    if p in (0.0, 1.0):
        return 0.0
    k = np.bincount(run_of, weights=succ, minlength=n_runs)
    n = np.bincount(run_of, minlength=n_runs)
    return float((((k - n * p) ** 2) / (n * p * (1 - p))).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs="+", action="append", required=True, metavar=("NAME", "RUN"),
                    help="a setting's name, then its runs (results_* directories or eval_results.json files)")
    ap.add_argument("--paper", default=None, help="the paper's successes/episodes for the cell, e.g. 22/25")
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    arms = {}
    print("| setting | runs | pooled | runs agree (permutation p) | median steps of successes |")
    print("|---|---|---|---|---|")
    for name, *runs in args.arm:
        if not runs:
            raise SystemExit(f"--arm {name}: no runs")
        eps = [load_episodes(r) for r in runs]
        succ = np.array([int(bool(e["success"])) for es in eps for e in es], float)
        run_of = np.repeat(np.arange(len(eps)), [len(es) for es in eps])
        p_runs = ""
        if len(eps) > 1:
            obs = chi2_over_runs(succ, run_of, len(eps))
            null = np.array([chi2_over_runs(rng.permutation(succ), run_of, len(eps)) for _ in range(args.permutations)])
            p_runs = f"{np.mean(null >= obs - 1e-12):.2f}"
        steps = [e["steps"] for es in eps for e in es if e["success"]]
        k, n = int(succ.sum()), len(succ)
        arms[name] = (k, n)
        print(f"| {name} | " + ", ".join(f"{sum(bool(e['success']) for e in es)}/{len(es)}" for es in eps)
              + f" | {rate(k, n)} | {p_runs} | {np.median(steps) if steps else float('nan'):.0f} |")

    print()
    for (a, (ka, na)), (b, (kb, nb)) in itertools.combinations(arms.items(), 2):
        print(f"{a} vs {b}: {ka / na:.0%} vs {kb / nb:.0%}, Fisher p = {fisher_two_sided(ka, na, kb, nb):.3f}")
    if args.paper:
        kp, n_paper = (int(v) for v in args.paper.split("/"))
        for a, (ka, na) in arms.items():
            print(f"{a} vs paper {kp}/{n_paper}: {ka / na:.0%} vs {kp / n_paper:.0%}, "
                  f"Fisher p = {fisher_two_sided(ka, na, kp, n_paper):.3f}")


if __name__ == "__main__":
    main()
