# SPDX-License-Identifier: Apache-2.0
"""How far each episode got: a task's milestones in order, pooled over repeat runs (README 30.14).

The eval stage records, per episode, which of the task's milestone predicates ever held (eval_results.json,
"milestones", in the task YAML's order). Two scenes with different success rates can then be compared step by
step. Each column counts the episodes that passed this step and every one before it, out of those that passed
the steps before:

    python repro/policy_eval_ours/milestones.py serve_fruits:gr00t:main,rep2,rep3 \\
        serve_fruits_ours_real:gr00t:assisted,assisted_rep2

Each argument is SCENE:POLICY:TAG[,TAG...], with POLICY the run folder's prefix (pi0, pi05, gr00t, gr00t17).
rollouts.hdf5 also stores milestones, but leaves some episodes empty, so this reads eval_results.json only.
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOTS = {"gr00t": "gr00t", "gr00t17": "gr00t", "pi05": "openpi_local", "pi0": "openpi_local"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="SCENE:POLICY:TAG[,TAG...]")
    ap.add_argument("--results", default=os.path.join(REPO, "results", "policy_eval_ours"))
    args = ap.parse_args()

    for spec in args.runs:
        scene, policy, tags = spec.split(":")
        episodes = []
        for tag in tags.split(","):
            found = sorted(glob.glob(os.path.join(args.results, ROOTS[policy], "out", scene, f"s15_eval_{policy}_{tag}",
                                                  "*", "*", "results_*", "eval_results.json")))
            if found:
                episodes += json.load(open(found[-1]))["episodes"]
        if not episodes:
            print(f"{spec}: no runs")
            continue
        names = list(episodes[0]["milestones"])
        passed = [True] * len(episodes)
        steps = []
        for name in names:
            before = sum(passed)
            passed = [p and bool(e["milestones"].get(name)) for p, e in zip(passed, episodes)]
            steps.append(f"{name} {sum(passed)}/{before}")
        successes = sum(bool(e["success"]) for e in episodes)
        print(f"{spec}: success {successes}/{len(episodes)} | " + " | ".join(steps))


if __name__ == "__main__":
    main()
