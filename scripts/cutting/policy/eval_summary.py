# SPDX-License-Identifier: Apache-2.0
"""Evaluation table: every seed counts (no demo filter), two success rates with Wilson 95 % intervals, failure kinds.

basic   cut_complete and >= 2 pieces and the second-largest piece >= 20 % of the block
strict  basic and largest force on the knife <= 3 N and the knife held (drift <= 5 deg) and no board strike
        (edge never > 1 mm under the board's collider top, board moved <= 2 mm)
Failure kinds (an episode can have several): missed (no cut started), incomplete (started, not through or a
sliver), force (> 3 N), knife (lost or turned > 5 deg in the hand), board_dig (edge > 1 mm under the board's
collider top), board_push (board moved > 2 mm), sim (the solver diverged; its wrench was capped before PhysX).
Every seed stays in the denominator, simulation failures included.
Usage: eval_summary.py <label>=<run dir> [...]   (labels go into the table as given)
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# (나) LiDAR scene: the in-sim board-top measure is not trusted there yet (NOTES 16:20), so its dig rule is left out.
IGNORE_DIG = __import__("os").environ.get("IGNORE_DIG") == "1"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score(j):
    o = json.loads(j.read_text())["outcome"]
    e = np.load(j.with_suffix(".npz"))
    fmax = float(np.linalg.norm(e["steps"][:, 1:4], axis=1).max())
    basic = bool(o["cut_complete"] and o["connected_pieces"] >= 2 and o["second_piece_frac"] >= 0.2)
    knife_ok = bool(o["held"] and o["hand_drift_deg"] <= 5.0)
    dig_ok = IGNORE_DIG or o["edge_min_above_board_mm"] >= -1.0
    board_ok = bool(dig_ok and o["board_moved_mm"] <= 2.0)
    strict = basic and fmax <= 3.0 and knife_ok and board_ok
    kinds = []
    if o.get("sim_blown"):
        kinds.append("sim")  # the solver diverged (capped before PhysX); counted as a failure of the episode
    if o["n_cuts"] == 0:
        kinds.append("missed")
    elif not basic:
        kinds.append("incomplete")
    if fmax > 3.0:
        kinds.append("force")
    if not knife_ok:
        kinds.append("knife")
    if not dig_ok:
        kinds.append("board_dig")
    if o["board_moved_mm"] > 2.0:
        kinds.append("board_push")
    # The edge's lowest point above the tofu's bottom face (from the block's own geometry, so it needs no board-top
    # measure; that one is wrong in the LiDAR scene). Runs before 15:30 recorded no edge height (6-column steps).
    st, blk = e["steps"], e["block"]
    edge_tofu = 1e3 * (float(np.nanmin(st[:, 6])) - float(blk[2] - blk[6] / 2)) if st.shape[1] >= 8 else None
    # The bottom phase: physics steps with the edge within 5 mm of the reference (board top; tofu bottom in the
    # LiDAR scene). The lowest point is an extreme value, so mere scatter of the commands near the floor makes it
    # deeper; the 10th percentile and the spread of the edge over this phase show the scatter itself (the edge
    # follows the command to ~0.1 mm, so this is the commanded height's spread).
    if IGNORE_DIG:
        h = 1e3 * (st[:, 6] - float(blk[2] - blk[6] / 2)) if st.shape[1] >= 8 else None
    else:
        h = st[:, 4]
    bottom = None
    if h is not None:
        b = h[np.isfinite(h) & (h < 5.0)]
        if len(b) >= 12:  # at least 0.1 s near the floor
            bottom = dict(p10=float(np.percentile(b, 10)), sd=float(b.std()), steps=int(len(b)))
    return dict(basic=basic, strict=strict, kinds=kinds, fmax=fmax, second=o["second_piece_frac"],
                edge_board=o["edge_min_above_board_mm"], edge_tofu=edge_tofu, bottom=bottom)


def order_in_process(j):
    """Where the episode ran in its process: recorded by the generator since 19:05, else the merged directory's
    process_order.json (process_order.py), else None."""
    info = json.loads(j.read_text())
    if "order_in_process" in info:
        return info["order_in_process"]
    side = j.parent / "process_order.json"
    if side.exists():
        hit = json.loads(side.read_text()).get(str(info["seed"]))
        return None if hit is None else hit["order"]
    return None


def start_turn_deg(paths):
    """Each episode's hand orientation at its first tick, in degrees from the run's most typical one. Before 19:05
    a process's first episode kept the setup grasp and started 30 deg off; this shows any such split, whatever
    causes it."""
    q = np.array([np.load(j.with_suffix(".npz"))["state"][0, 12:16].astype(np.float64) for j in paths])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    a = np.degrees(2 * np.arccos(np.clip(np.abs(q @ q.T), 0.0, 1.0)))
    return a[int(np.argmin(np.median(a, axis=1)))]


def depth(sc, n_boot=10000):
    """Median and mean of each episode's lowest edge height [mm] with bootstrap 95 % intervals over the seeds.

    The strict count flips on a quarter of the episodes between identical runs (edges a fraction of a mm either side
    of -1 mm); these move less. Reference: the board's collider top, or in the LiDAR scene (IGNORE_DIG) the tofu's
    bottom face."""
    key = "edge_tofu" if IGNORE_DIG else "edge_board"
    x = np.array([s[key] for s in sc], dtype=float) if all(s[key] is not None for s in sc) else None
    if x is None or len(x) == 0:
        return None
    b = np.random.default_rng(0).choice(x, size=(n_boot, len(x)), replace=True)
    med, mean = np.median(b, axis=1), b.mean(axis=1)
    return dict(ref="두부 바닥" if IGNORE_DIG else "도마 윗면", median=float(np.median(x)), mean=float(x.mean()),
                median_boot95=[float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))],
                mean_boot95=[float(np.percentile(mean, 2.5)), float(np.percentile(mean, 97.5))])


def bottom_phase(sc, n_boot=10000):
    """Over the episodes that reached the bottom phase: the median of their 10th percentiles and of their spreads
    [mm], with bootstrap 95 % intervals."""
    b = [s["bottom"] for s in sc if s.get("bottom")]
    if not b:
        return None
    rng = np.random.default_rng(0)
    out = dict(episodes=len(b))
    for k in ("p10", "sd"):
        x = np.array([v[k] for v in b])
        m = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
        out[k] = float(np.median(x))
        out[k + "_boot95"] = [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    return out


def tally(sc):
    n = len(sc)
    kb, ks = sum(s["basic"] for s in sc), sum(s["strict"] for s in sc)
    kinds = {}
    for s in sc:
        for k in s["kinds"]:
            kinds[k] = kinds.get(k, 0) + 1
    return dict(n=n, basic=kb, strict=ks, basic_wilson95=wilson(kb, n), strict_wilson95=wilson(ks, n), kinds=kinds,
                edge=depth(sc), bottom=bottom_phase(sc))


def run_row(d):
    paths = sorted(Path(d).glob("ep_*.json"))
    sc = [score(j) for j in paths]
    seeds = [json.loads(j.read_text())["seed"] for j in paths]
    order = [order_in_process(j) for j in paths]
    turn = start_turn_deg(paths)
    r = dict(dir=str(d), **tally(sc), order_known=all(o is not None for o in order),
             first_seeds=[s for s, o in zip(seeds, order) if o == 0],
             turned_start_seeds=[s for s, t in zip(seeds, turn) if t > 10.0])
    if r["order_known"]:
        r["excl_first"] = tally([s for s, o in zip(sc, order) if o != 0])
    return r


def report(label, r):
    wb, ws = r["basic_wilson95"], r["strict_wilson95"]
    print(f"{label:34s} {r['n']:3d}  {r['basic']:3d} [{100 * wb[0]:4.0f},{100 * wb[1]:4.0f}]%  {r['strict']:3d} "
          f"[{100 * ws[0]:4.0f},{100 * ws[1]:4.0f}]%  {r['kinds']}  first {r['first_seeds']} "
          f"turned {r['turned_start_seeds']}")


def main():
    """<label>=<dir> gives one row; <label>=<dir>+<dir> gives one row per run of the same evaluation (same checkpoint,
    seeds and guards), which results_table.py follows with their repeat range."""
    rows = {}
    print(f"{'run':34s} {'n':>3s}  {'basic':>17s}  {'strict':>17s}  failure kinds")
    for arg in sys.argv[1:]:
        label, ds = arg.split("=", 1)
        runs = ds.split("+")
        for i, d in enumerate(runs):
            r = run_row(d)
            name = label if len(runs) == 1 else f"{label} ({i + 1}회차)"
            if len(runs) > 1:
                r["repeat"] = dict(group=label, index=i + 1, of=len(runs))
            rows[name] = r
            report(name, r)
    out = Path(os.environ.get("EVAL_SUMMARY_OUT", "eval_summary.json"))  # where the table tools read it
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
