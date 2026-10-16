# SPDX-License-Identifier: Apache-2.0
"""Fixed episode starts for the Table G.1 evaluation (repro/README.md, chapter 18).

Paper App. J.1: "each of the objects has a defined spatial reset distribution ... uniformly divided into a
5-by-5 grid, yielding 25 positions per object. For each rollout, one of the 25 positions is sampled
independently per object, without replacement ... We also sample a rotation for each object per position,
and these positions are held fixed across all checkpoints." The repository ships the configuration schema
of the tool that did this (scripts/cfg/sample_init_states/example.yaml) but not the tool. This follows that
schema:

    grid_size        full extent of the grid, centred on the object's position in the scene
    z_rot_range      yaw added to the scene orientation, linearly spaced over the 25 grid cells
    randomize_pairings / pairing_anchor_object
                     each object takes its own shuffled order of cells; the anchor keeps the canonical one
    n_physics_steps, max_displacement_threshold, max_velocity_threshold, max_retries, retry_perturbation
                     a start is kept only if every listed object stays put in physics; a failing one is
                     retried with small XY perturbations

Where perturbations are not enough (two grids that overlap by more than an object's width), an object's
cells are swapped between two episodes and both are checked again, so each object still visits each of its
25 cells exactly once. The output is what 1_eval_policy_og_scene.py reads through s15_eval.init_states_path.

Runs in the simfoundry env (boots Isaac Sim), one task per process:
    python repro/policy_eval/sample_init_states.py --task marker_in_cup --out /content/eval/init_states/marker_in_cup.json
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = {
    "n_physics_steps": 30,
    "max_velocity_threshold": 0.05,
    "max_displacement_threshold": 0.02,
    "max_retries": 10,
    "retry_perturbation": 0.02,
    "max_swaps": 400,
}


def grid_cells(spec, res):
    """(dx, dy, yaw) of each of the nx*ny cells, row-major; rows run along x."""
    nx, ny = res
    sx, sy = spec["grid_size"]
    xs = np.linspace(-sx / 2, sx / 2, nx) if nx > 1 else np.zeros(1)
    ys = np.linspace(-sy / 2, sy / 2, ny) if ny > 1 else np.zeros(1)
    lo, hi = spec.get("z_rot_range", [0.0, 0.0])
    yaws = np.linspace(lo, hi, nx * ny)
    return [(float(xs[i // ny]), float(ys[i % ny]), float(yaws[i])) for i in range(nx * ny)]


def yaw_quat(yaw):
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


def quat_mul(a, b):
    """Hamilton product of xyzw quaternions: a * b (b applied first)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


class Starts:
    """The 25 starts: which cell each object takes in each episode, plus any retry perturbation."""

    def __init__(self, cfg, default_poses):
        self.res = cfg["grid_resolution"]
        self.n = self.res[0] * self.res[1]
        self.objects = [o["name"] for o in cfg["objects"]]
        self.cells = {o["name"]: grid_cells(o, self.res) for o in cfg["objects"]}
        self.default = default_poses
        self.anchor = cfg.get("pairing_anchor_object")
        self.rng = np.random.default_rng(cfg.get("seed", 42))
        self.assign = {}
        for name in self.objects:
            if cfg.get("randomize_pairings", True) and name != self.anchor:
                self.assign[name] = self.rng.permutation(self.n)
            else:
                self.assign[name] = np.arange(self.n)
        self.jitter = {k: {name: np.zeros(2) for name in self.objects} for k in range(self.n)}

    def poses(self, k):
        out = {}
        for name in self.objects:
            dx, dy, yaw = self.cells[name][int(self.assign[name][k])]
            pos, ori = self.default[name]
            p = np.array(pos, dtype=float)
            p[:2] += np.array([dx, dy]) + self.jitter[k][name]
            q = quat_mul(yaw_quat(yaw), np.array(ori, dtype=float))
            out[name] = (p, q / np.linalg.norm(q))
        return out

    def record(self, k, check):
        anchor = self.anchor or self.objects[0]
        cell = int(self.assign[anchor][k])
        state = {
            "state_idx": k,
            "grid_row": cell // self.res[1],
            "grid_col": cell % self.res[1],
            "cells": {name: int(self.assign[name][k]) for name in self.objects},
            "objects": {},
        }
        for name, (p, q) in self.poses(k).items():
            state["objects"][name] = {"pos": [round(float(v), 6) for v in p], "ori": [round(float(v), 6) for v in q]}
        if check is not None:
            state["check"] = check
        return state


def default_poses(scene, names):
    reg = scene["state"]["registry"]["object_registry"]
    missing = [n for n in names if n not in reg]
    if missing:
        sys.exit(f"objects not in the scene: {missing}; it has {sorted(reg)}")
    return {n: (reg[n]["root_link"]["pos"], reg[n]["root_link"]["ori"]) for n in names}


class PhysicsCheck:
    """Places a start in the eval's own environment and reports how far each object moves."""

    def __init__(self, scene_path, task_yaml, resolved_path, params):
        import omnigibson as og
        from omnigibson.macros import gm
        from omnigibson.utils.config_utils import parse_config
        from simfoundry import import_og_dependencies
        from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

        gm.ENABLE_CCD = True  # as in 1_eval_policy_og_scene.py
        import_og_dependencies()
        self.og = og
        self.p = params
        scene_json = load_json_with_absolute_usd_paths(scene_path)
        # One file per task: tasks sampled side by side once shared a name here, and one of them loaded
        # the other's scene.
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        with open(resolved_path, "w") as f:
            json.dump(scene_json, f)
        task_cfg = parse_config(task_yaml)["og_task_config"]
        scene_cfg = {"type": "Scene", "scene_file": resolved_path, "use_floor_plane": True,
                     "floor_plane_visible": False, "use_skybox": False, "include_robots": True}
        env_cfg = {"action_frequency": 15, "rendering_frequency": 15, "physics_frequency": 120}
        self.env = og.Environment(configs=dict(env=env_cfg, scene=scene_cfg, task=task_cfg))
        self.env.reset()
        # Every check starts from the state the eval's reset leaves (robot at its reset pose, objects at the
        # scene's), restored directly: env.reset() itself retries until objects are still, about 5 s a time.
        self.reset_state = og.sim.dump_state(serialized=True)

    def __call__(self, poses):
        import torch as th

        self.og.sim.load_state(self.reset_state, serialized=True)
        objs = {}
        for name, (p, q) in poses.items():
            obj = self.env.scene.object_registry("name", name)
            obj.set_position_orientation(position=th.tensor(p, dtype=th.float32), orientation=th.tensor(q, dtype=th.float32))
            obj.keep_still()
            objs[name] = obj
        for _ in range(self.p["n_physics_steps"]):
            self.og.sim.step_physics()
        moved = {}
        for name, obj in objs.items():
            pos, _ = obj.get_position_orientation()
            disp = float(np.linalg.norm(pos.cpu().numpy() - poses[name][0]))
            vel = float(th.linalg.norm(obj.get_linear_velocity()).item())
            moved[name] = {"displacement": round(disp, 4), "velocity": round(vel, 4)}
        ok = all(m["displacement"] <= self.p["max_displacement_threshold"]
                 and m["velocity"] <= self.p["max_velocity_threshold"] for m in moved.values())
        return ok, moved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=os.path.join(HERE, "table_g1.yaml"))
    ap.add_argument("--task", required=True, help="key under tasks: in the spec")
    ap.add_argument("--out", required=True)
    ap.add_argument("--asset-dir", default=None, help="default: simfoundry.ASSET_DIR")
    ap.add_argument("--task-yaml", default=None, help="default: repro/policy_eval/tasks/<task>.yaml")
    ap.add_argument("--no-physics", action="store_true", help="grid and pairings only, no Isaac Sim")
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.spec))["tasks"][args.task]
    cfg = spec["init_states"]
    params = {**DEFAULTS, **{k: cfg[k] for k in DEFAULTS if k in cfg}}
    if args.asset_dir is None:
        from simfoundry import ASSET_DIR as asset_dir
    else:
        asset_dir = args.asset_dir
    scene_path = os.path.join(asset_dir, spec["scene"])
    scene = json.load(open(scene_path))
    starts = Starts(cfg, default_poses(scene, [o["name"] for o in cfg["objects"]]))
    checks = {}

    if not args.no_physics:
        task_yaml = args.task_yaml or os.path.join(HERE, "tasks", f"{spec['task']}.yaml")
        resolved = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"{args.task}_scene_resolved.json")
        check = PhysicsCheck(scene_path, task_yaml, resolved, params)

        def try_state(k):
            """Check start k, retrying with jitter. True once it holds."""
            starts.jitter[k] = {name: np.zeros(2) for name in starts.objects}
            for attempt in range(params["max_retries"] + 1):
                ok, moved = check(starts.poses(k))
                checks[k] = {"moved": moved, "retries": attempt}
                worst = max(moved, key=lambda n: moved[n]["displacement"])
                print(f"[init_states] start {k} try {attempt}: {'ok' if ok else 'unstable'} "
                      f"(most moved: {worst} {moved[worst]['displacement']:.3f} m, {moved[worst]['velocity']:.3f} m/s)",
                      flush=True)
                if ok:
                    return True
                for name in starts.objects:
                    if name == starts.anchor:
                        continue
                    r = params["retry_perturbation"] * np.sqrt(starts.rng.random())
                    a = 2 * np.pi * starts.rng.random()
                    starts.jitter[k][name] = np.array([r * np.cos(a), r * np.sin(a)])
            return False

        bad = [k for k in range(starts.n) if not try_state(k)]
        swaps = 0
        while bad and swaps < params["max_swaps"]:
            k = bad[0]
            moved = checks[k]["moved"]
            movable = [n for n in starts.objects if n != starts.anchor] or starts.objects
            name = max(movable, key=lambda n: moved[n]["displacement"])
            j = int(starts.rng.choice([i for i in range(starts.n) if i != k]))
            a = starts.assign[name]
            a[k], a[j] = a[j], a[k]
            swaps += 1
            ok_k, ok_j = try_state(k), try_state(j)
            if ok_k and ok_j:
                bad.remove(k)
                if j in bad:
                    bad.remove(j)
                print(f"[init_states] start {k}: swapped {name}'s cell with start {j}", flush=True)
            else:
                a[k], a[j] = a[j], a[k]
                if not try_state(j) and j not in bad:
                    bad.append(j)
        if bad:
            sys.exit(f"[init_states] {len(bad)} starts still unstable after {swaps} swaps: {bad}")

    out = {
        "task": args.task,
        "scene": spec["scene"],
        "config": cfg,
        "physics_check": None if args.no_physics else params,
        "init_states": [starts.record(k, checks.get(k)) for k in range(starts.n)],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    retries = sum(c["retries"] > 0 for c in checks.values())
    print(f"[init_states] {args.task}: {starts.n} starts -> {args.out} ({retries} needed a perturbation)", flush=True)
    if not args.no_physics:
        import omnigibson as og
        og.shutdown()


if __name__ == "__main__":
    main()
