# SPDX-License-Identifier: Apache-2.0
"""Collect the Table G.1 runs (repro/README.md, chapter 18) and compare them with the paper.

Reads every eval_results.json under <root>/<task>/s15_eval_<policy>_<tag>/ (written by
1_eval_policy_og_scene.py), takes the newest per task and policy, and prints a Markdown table: successes
out of episodes with a Wilson 95% interval, the paper's simulation and real-world rates, Fisher's exact
two-sided p for "same success probability as the paper's simulation run" (the paper ran 25 episodes per cell
too, App. J.1), per-milestone rates, and mean episode length.

    python repro/policy_eval/summarize.py --root /content/eval/out --tag main --json summary.json
"""
import argparse
import glob
import json
import math
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fisher_two_sided(k1, n1, k2, n2):
    """Two-sided Fisher exact test on [[k1, n1-k1], [k2, n2-k2]]."""
    K, N = k1 + k2, n1 + n2

    def pmf(x):
        return math.comb(K, x) * math.comb(N - K, n1 - x) / math.comb(N, n1)

    lo, hi = max(0, K - n2), min(K, n1)
    p_obs = pmf(k1)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-9)))


def latest_results(root, task, policy, tag):
    paths = glob.glob(os.path.join(root, task, f"s15_eval_{policy}_{tag}", "*", "*", "results_*", "eval_results.json"))
    return max(paths, key=os.path.getmtime) if paths else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--spec", default=os.path.join(HERE, "table_g1.yaml"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    spec = yaml.safe_load(open(args.spec))

    rows, out = [], []
    for task, ts in spec["tasks"].items():
        for policy in spec["policies"]:
            path = latest_results(args.root, task, policy, args.tag)
            if path is None:
                rows.append(f"| {task} | {policy} | (no results) | | | | | | |")
                continue
            r = json.load(open(path))
            eps = r["episodes"]
            n = len(eps)
            k = sum(bool(e["success"]) for e in eps)
            lo, hi = wilson(k, n)
            paper = ts.get("paper", {}).get(policy)
            real = ts.get("paper_real", {}).get(policy)
            base = ts.get("paper_base", {}).get(policy)
            fisher = None
            if paper is not None:
                kp = round(paper / 100 * 25)
                fisher = fisher_two_sided(k, n, kp, 25)
            milestones = {}
            for e in eps:
                for name, hit in (e.get("milestones") or {}).items():
                    milestones.setdefault(name, 0)
                    milestones[name] += int(bool(hit))
            ms = ", ".join(f"{name} {v}/{n}" for name, v in milestones.items())
            steps = sum(e["steps"] for e in eps) / max(n, 1)
            ref = f"{paper}" + (f" ({base} base)" if base is not None else "")
            rows.append(
                f"| {task} | {policy} | {k}/{n} = {100 * k / max(n, 1):.0f}% | {100 * lo:.0f}-{100 * hi:.0f} | "
                f"{ref} | {real if real is not None else ''} | "
                f"{'' if fisher is None else f'{fisher:.2f}'} | {ms} | {steps:.0f} |")
            out.append({"task": task, "policy": policy, "episodes": n, "successes": k, "rate": k / max(n, 1),
                        "wilson95": [lo, hi], "paper_sim": paper, "paper_real": real, "paper_base": base,
                        "fisher_p_vs_paper_sim": fisher, "milestones": milestones, "mean_steps": steps,
                        "prompt": r.get("prompt"), "execute_horizon": r.get("execute_horizon"),
                        "n_steps": r.get("n_steps"), "results": path})
    print("| task | policy | success | 95% CI (%) | paper sim (%) | paper real (%) | Fisher p | milestones | mean steps |")
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
