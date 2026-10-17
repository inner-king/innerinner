# SPDX-License-Identifier: Apache-2.0
"""Fixed episode starts drawn by a task config's own reset randomization, for a scene whose authors' grid is unknown.

For Cup in Bowl the authors released neither the scene nor its evaluation grid (paper App. J.1). The repository's
task config (scripts/cfg/task/droid/droid_desk_put_cup_in_bowl.yaml) does say how the objects are spread: per-group
xyz offsets, workspace bounds in the robot's frame, and the bowl placed next to the cup on one of four sides
(group_predicate_placement). This resets an environment with that config N times and keeps each reset that is a
valid start, so the starts come from the authors' own placement code instead of a layout guessed here:

  - the task's randomization runs on env.reset() (PickPlaceTask._reset_scene), with Python's, NumPy's and torch's
    generators seeded, so a seed gives the same starts;
  - physics then runs --settle-steps steps; a reset is kept only if no object moved more than --max-move and the
    task's milestones are all still unmet (the cup is not already in the bowl);
  - the poses kept are the settled ones, in the init-state format the eval stage reads
    (s15_eval.init_states_path), with each start's bowl side relative to the cup recorded for reference.

Scene masses are applied as the eval stage applies them, so the physics check sees the masses the runs use.

Runs in the simfoundry env (boots Isaac Sim):
    python repro/policy_eval_ours/sample_task_starts.py --task cup_in_bowl_ours_cfg \\
        --out results/policy_eval_ours/gr00t/init_states/cup_in_bowl_ours_cfg.json
"""
import argparse
import json
import os
import random

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def side_of(cup_xy, bowl_xy, robot_xy):
    """Where the bowl is relative to the cup, seen from the robot (x forward, y left)."""
    dx, dy = np.asarray(bowl_xy) - np.asarray(cup_xy)
    if abs(dx) >= abs(dy):
        return "behind" if dx > 0 else "in_front_of"
    return "left_of" if dy > 0 else "right_of"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=os.path.join(HERE, "ours.yaml"))
    ap.add_argument("--task", required=True, help="key under tasks: in the spec (its init_states.task_yaml is used)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--max-tries", type=int, default=200)
    ap.add_argument("--settle-steps", type=int, default=30)
    ap.add_argument("--max-move", type=float, default=0.02, help="largest displacement while settling (m)")
    args = ap.parse_args()

    import omnigibson as og
    import torch as th
    from omnigibson.macros import gm
    from omnigibson.utils.config_utils import parse_config
    from simfoundry import ASSET_DIR, import_og_dependencies
    from simfoundry.utils.og_utils import apply_scene_masses
    from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

    gm.ENABLE_CCD = True  # as in the eval stage
    import_og_dependencies()
    spec = yaml.safe_load(open(args.spec))["tasks"][args.task]
    starts_cfg = spec["init_states"]
    seed = int(starts_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    work = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(work, exist_ok=True)
    resolved = os.path.join(work, f"{args.task}_scene_resolved.json")
    scene_json = load_json_with_absolute_usd_paths(os.path.join(ASSET_DIR, spec["scene"]))
    with open(resolved, "w") as f:
        json.dump(scene_json, f)
    task_cfg = parse_config(os.path.join(REPO, starts_cfg["task_yaml"]))["og_task_config"]
    env = og.Environment(configs=dict(
        env={"action_frequency": 15, "rendering_frequency": 15, "physics_frequency": 120},
        scene={"type": "Scene", "scene_file": resolved, "use_floor_plane": True, "floor_plane_visible": False,
               "use_skybox": False, "include_robots": True},
        task=task_cfg))
    with og.sim.stopped():
        apply_scene_masses(env, scene_json)

    names = [o["name"] for o in starts_cfg["objects"]]
    robot_xy = np.asarray(env.robots[0].get_position_orientation()[0].cpu())[:2]
    starts, tries = [], 0
    while len(starts) < args.n and tries < args.max_tries:
        tries += 1
        env.reset()
        objs = {n: env.scene.object_registry("name", n) for n in names}
        placed = {n: np.asarray(o.get_position_orientation()[0].cpu()) for n, o in objs.items()}
        for _ in range(args.settle_steps):
            og.sim.step_physics()
        settled = {n: o.get_position_orientation() for n, o in objs.items()}
        moved = {n: float(np.linalg.norm(np.asarray(settled[n][0].cpu()) - placed[n])) for n in names}
        milestones = env.task.check_milestones(env)
        reached = [k for k, v in dict(milestones).items() if v]
        if max(moved.values()) > args.max_move or reached:
            print(f"[starts] try {tries}: rejected (moved {max(moved.values()) * 100:.1f} cm, met {reached})")
            continue
        cup, bowl = (n for n in names if "cup" in n), (n for n in names if "bowl" in n)
        cup, bowl = next(cup, None), next(bowl, None)
        side = side_of(placed[cup][:2], placed[bowl][:2], robot_xy) if cup and bowl else None
        starts.append({"state_idx": len(starts), "try": tries, "bowl_side": side,
                       "objects": {n: {"pos": [round(float(x), 6) for x in settled[n][0].cpu()],
                                       "ori": [round(float(x), 6) for x in settled[n][1].cpu()]} for n in names},
                       "check": {"moved_m": {n: round(m, 4) for n, m in moved.items()}}})
        print(f"[starts] try {tries}: start {len(starts) - 1}, bowl {side} of the cup")
    out = {"task": args.task, "scene": spec["scene"], "sampler": "task_config", "task_yaml": starts_cfg["task_yaml"],
           "seed": seed, "tries": tries, "settle_steps": args.settle_steps, "max_move_m": args.max_move,
           "init_states": starts}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    sides = {s: sum(x["bowl_side"] == s for x in starts) for s in ("left_of", "right_of", "behind", "in_front_of")}
    print(f"[starts] {args.task}: {len(starts)} starts from {tries} resets -> {args.out}; bowl sides {sides}")
    og.shutdown()


if __name__ == "__main__":
    main()
