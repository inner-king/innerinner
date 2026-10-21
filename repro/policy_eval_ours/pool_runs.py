# SPDX-License-Identifier: Apache-2.0
"""Pool repeat runs of the same cell into one success count, in the list correlate.py --summary reads.

Each cell of README 30.10–30.14 is two or more 25-episode runs under different tags (assisted, assisted_rep2, ...),
run on this machine and on super2 and copied into one results tree. This adds them up per task and policy:

    python repro/policy_eval_ours/pool_runs.py OUT.json TASK[,TASK...] [TAG[,TAG...]]
    python repro/policy_eval_ours/correlate.py --summary OUT.json --draws 20000

TAGs default to main,rep2,rep3,rep4. A run counts only when it finished all 25 episodes; when one tag holds several
results folders, the newest is read, as correlate.py does. The per-run counts are printed before the pooled list.
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# Folder prefix of each policy's runs (s15_eval_<prefix>_<tag>) -> the eval root it is written under.
ROOTS = {"gr00t": "gr00t", "gr00t17": "gr00t", "pi05": "openpi_local", "pi0": "openpi_local"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("tasks", help="comma-separated scene names, e.g. cup_in_bowl_ours_real,serve_fruits_ours_real")
    ap.add_argument("tags", nargs="?", default="main,rep2,rep3,rep4")
    ap.add_argument("--results", default=os.path.join(REPO, "results", "policy_eval_ours"))
    ap.add_argument("--episodes", type=int, default=25, help="a run with fewer episodes is left out")
    args = ap.parse_args()

    rows, per_run = [], {}
    for task in args.tasks.split(","):
        for policy, root in ROOTS.items():
            successes = episodes = 0
            for tag in args.tags.split(","):
                found = sorted(glob.glob(os.path.join(args.results, root, "out", task, f"s15_eval_{policy}_{tag}",
                                                      "*", "*", "results_*", "eval_results.json")))
                if not found:
                    continue
                eps = json.load(open(found[-1]))["episodes"]
                if len(eps) < args.episodes:
                    continue
                k = sum(bool(e["success"]) for e in eps)
                per_run[f"{task}/{policy}_{tag}"] = [k, len(eps)]
                successes += k
                episodes += len(eps)
            if episodes:
                rows.append({"task": task, "policy": policy, "successes": successes, "episodes": episodes})
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(json.dumps(per_run, indent=0))
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
