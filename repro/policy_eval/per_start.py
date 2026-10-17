# SPDX-License-Identifier: Apache-2.0
"""Which fixed starts of a Table G.1 task fail, over several runs on the same starts (repro/README.md, 18.4).

Each run of 1_eval_policy_og_scene.py with s15_eval.init_states_path plays episode k from start k, so runs
on one init-state file can be lined up start by start. This prints, per start, the cell of each object's
5x5 grid, the distance between the object the robot picks and the one it places into, and the successes
over the runs; then whether the starts differ at all (a permutation test of "every start has the same
success probability"), the rank correlation of distance and successes over starts, the success rate by
distance half and by the picked object's inner 3x3 cells against its outer ring (Wilson 95% intervals,
Fisher's exact test), and the counts by grid row and column.

    python repro/policy_eval/per_start.py --init repro/policy_eval/init_states/marker_in_cup.json \
        --runs out/marker_in_cup/s15_eval_pi05_{main,mug300g,calcam,prompt_repo}/*/*/results_*
"""
import argparse
import json
import os

import numpy as np

from summarize import fisher_two_sided, wilson


def load_successes(run):
    path = run if run.endswith(".json") else os.path.join(run, "eval_results.json")
    eps = json.load(open(path))["episodes"]
    return {int(e["episode_idx"]): int(bool(e["success"])) for e in eps}


def rate(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} ({lo:.0%}-{hi:.0%})"


def ranks(x):
    """Average ranks (ties share the mean rank), for Spearman's rho."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="init-state file the runs used")
    ap.add_argument("--runs", nargs="+", required=True, help="results_* directories or eval_results.json files")
    ap.add_argument("--pick", default=None, help="object the robot picks (default: the first in the grid config)")
    ap.add_argument("--place", default=None, help="object it places into (default: the second)")
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    init = json.load(open(args.init))
    names = [o["name"] for o in init["config"]["objects"]]
    pick, place = args.pick or names[0], args.place or names[1]
    ny = init["config"]["grid_resolution"][1]
    starts = init["init_states"]
    n = len(starts)
    runs = [load_successes(r) for r in args.runs]
    for r, path in zip(runs, args.runs):
        if sorted(r) != list(range(n)):
            raise SystemExit(f"{path}: episodes {sorted(r)} do not cover the {n} starts")
    S = np.array([[r[k] for r in runs] for k in range(n)])  # starts x runs
    m = S.shape[1]
    succ = S.sum(1)

    pick_xy = np.array([s["objects"][pick]["pos"][:2] for s in starts])
    place_xy = np.array([s["objects"][place]["pos"][:2] for s in starts])
    dist = np.linalg.norm(pick_xy - place_xy, axis=1)
    pcell = np.array([s["cells"][pick] for s in starts])
    qcell = np.array([s["cells"][place] for s in starts])

    print(f"{n} starts x {m} runs; {pick} -> {place}; overall {rate(int(succ.sum()), n * m)}\n")
    print("| start | pick cell (row, col) | place cell | distance (m) | " + " | ".join(f"run {i + 1}" for i in range(m))
          + " | successes |")
    print("|---|---|---|---|" + "---|" * m + "---|")
    for k in np.argsort(dist, kind="mergesort"):
        print(f"| {k} | ({pcell[k] // ny}, {pcell[k] % ny}) | ({qcell[k] // ny}, {qcell[k] % ny}) | {dist[k]:.3f} | "
              + " | ".join(str(v) for v in S[k]) + f" | {succ[k]}/{m} |")

    rng = np.random.default_rng(args.seed)
    flat = S.flatten()

    def spread(f):
        return float(((f.reshape(n, m).sum(1) - f.mean() * m) ** 2).sum())

    obs = spread(flat)
    p_same = float(np.mean([spread(rng.permutation(flat)) >= obs for _ in range(args.permutations)]))
    print(f"\nSame success probability at every start: permutation p = {p_same:.4f} "
          f"(starts with 0..{m} successes: {np.bincount(succ, minlength=m + 1).tolist()})")

    rd = ranks(dist)
    rho = float(np.corrcoef(rd, ranks(succ))[0, 1])
    null = np.array([np.corrcoef(rd, ranks(rng.permutation(succ)))[0, 1] for _ in range(args.permutations)])
    print(f"Distance vs successes per start: Spearman rho = {rho:.2f}, permutation p = {np.mean(np.abs(null) >= abs(rho)):.3f}")

    order = np.argsort(dist, kind="mergesort")
    near, far = order[: n // 2], order[n - n // 2:]
    kn, kf = int(succ[near].sum()), int(succ[far].sum())
    print(f"Nearer half ({dist[near].min():.2f}-{dist[near].max():.2f} m): {rate(kn, m * len(near))}; "
          f"farther half ({dist[far].min():.2f}-{dist[far].max():.2f} m): {rate(kf, m * len(far))}; "
          f"Fisher p = {fisher_two_sided(kn, m * len(near), kf, m * len(far)):.3f} (episodes, not starts, as units)")

    nx = init["config"]["grid_resolution"][0]
    row, col = pcell // ny, pcell % ny
    inner = (row > 0) & (row < nx - 1) & (col > 0) & (col < ny - 1)
    ki, ko = int(succ[inner].sum()), int(succ[~inner].sum())
    print(f"{pick} inner {nx - 2}x{ny - 2} cells: {rate(ki, m * inner.sum())}; outer ring: {rate(ko, m * (~inner).sum())}; "
          f"Fisher p = {fisher_two_sided(ki, m * inner.sum(), ko, m * (~inner).sum()):.3f}")
    for obj, cells in ((pick, pcell), (place, qcell)):
        for axis, v in (("row (x)", cells // ny), ("col (y)", cells % ny)):
            print(f"{obj} by {axis}: " + ", ".join(f"{int(succ[v == i].sum())}/{m * int((v == i).sum())}"
                                                   for i in range(nx if axis.startswith("row") else ny)))


if __name__ == "__main__":
    main()
