# SPDX-License-Identifier: Apache-2.0
"""Replay B7 (task proposal) at many seeds on two scenes, to tell the scene's effect from the draw's.

The end-to-end run with Qwen3.5 as Pipeline A's VLM (repro/README.md, chapter 16) proposed
pick_up_marker and place_scissors_in_cup on put_marker_in_cup_q35, while the all-Qwen3-VL v2 scene
had proposed the video's task, the marker into the cup. B7 samples at temperature 0.7 and the stage
always asks with seed 0, so one run per scene cannot say whether the scene or the draw decided it.

This sends the stage's call the way Pipeline B sends it (run.sh passes propose_scene_task.num_tasks=2)
through Gemini() -> shim -> Ollama, with replay_other_calls.run_b7 (the stage's own prompt builder and
parser), at seeds 0..N-1 on each scene. It counts the answers whose written tasks put the marker
inside the cup (an InsideAABB predicate on the objects the groups map to) and the tasks that parse.
Seed 0 is the pipeline's own call, so its tasks are checked against the YAMLs the run wrote.

Nothing is written under Data/. Run with the response cache off and Pipeline B's shim settings:
  SIMFOUNDRY_LOCAL_BACKEND=1 SIMFOUNDRY_OLLAMA_MODEL=qwen3-vl:8b-instruct SIMFOUNDRY_OLLAMA_THINK=0
  SIMFOUNDRY_OLLAMA_KEEP_ALIVE=0 GCLOUD_PROJECT=local

Usage: python repro/vlm_compare/replay_b7_seeds.py <out_dir> [--seeds N] [--repeat 0,1] [--data-root DIR]
A rerun with the same <out_dir> skips the (scene, seed) pairs already answered; --summarize-only
grades the calls in <out_dir> again without making any.
"""
import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc  # noqa: E402  (puts the repository root on sys.path)
import replay_other_calls as roc  # noqa: E402

import pynvml  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

SCENES = ("put_marker_in_cup_v2", "put_marker_in_cup_q35")
RUN_SH = bc.REPO / "scripts/pipeline/B_augmentation/run.sh"
# Pipeline B asks for two tasks rather than the config's five (replay_other_calls.py asks for five),
# and the stage writes only the first num_tasks it parses.
NUM_TASKS = 2
assert f'"propose_scene_task.num_tasks={NUM_TASKS}"' in RUN_SH.read_text(), f"{RUN_SH} changed num_tasks"
roc.CFG.propose_scene_task.num_tasks = NUM_TASKS
# Attempts per (scene, seed) before giving up on getting one that no other job overlapped.
MAX_TRIES = 6


# ---------------------------------------------------------------------------------------------- GPU
def _cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
    except OSError:
        return ""


def _running_cmdline(pid):
    """Command line of a process that is still running and is not Ollama's; None otherwise.

    Every load (keep_alive 0 loads and unloads on each call) briefly puts Ollama children on the GPU
    whose command line reads empty while NVML still lists them: the device probes
    (`llama-server --list-devices`, `ollama gpu-discover`) as zombies, and the runner as it exits.
    bench_common counted those as another job, and since a seed's answer and timing repeat exactly,
    the same call kept being discarded. The parent still names Ollama, so that is checked too.
    """
    try:
        state, ppid = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[:2]
    except (OSError, ValueError):
        return None  # gone
    cmd = _cmdline(pid)
    if state in ("Z", "X") or "ollama" in cmd or "ollama" in _cmdline(int(ppid)):
        return None
    return cmd


def _comfy_idle():
    try:
        q = bc.requests.get(f"{bc.lb.COMFY_URL}/queue", timeout=2).json()
    except Exception:
        return False
    return not q.get("queue_running") and not q.get("queue_pending")


def other_gpu_jobs():
    """bench_common.other_gpu_jobs, with three changes.

    Ollama's short-lived children are not jobs (see _running_cmdline). An idle ComfyUI (models unloaded,
    empty queue) is not one either: Pipeline B leaves one running, so the pipeline's own B7 calls were
    made beside it, and bench_common would wait on it forever. And a graphics-only client counts when
    it is Python or Kit, so an Isaac Sim render is caught even if it holds no CUDA context.
    """
    pids = []
    for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(bc._GPU):
        cmd = _running_cmdline(proc.pid)
        if cmd is None:
            continue
        if "comfyui" in cmd.lower() and (proc.usedGpuMemory or 0) < 512 * 2**20 and _comfy_idle():
            continue
        pids.append(proc.pid)
    for proc in pynvml.nvmlDeviceGetGraphicsRunningProcesses(bc._GPU):
        cmd = _running_cmdline(proc.pid)
        if cmd is None or proc.pid in pids or "comfyui" in cmd.lower():
            continue
        if "python" in cmd or re.search(r"(^|/)kit(\s|$)", cmd) or "isaac" in cmd.lower():
            pids.append(proc.pid)
    return pids


def wait_for_idle_gpu():
    if other_gpu_jobs():
        t0 = time.time()
        print(f"   waiting for the GPU: {[(p, _cmdline(p)[:80]) for p in other_gpu_jobs()]}", flush=True)
        while other_gpu_jobs():
            time.sleep(10)
        print(f"   GPU idle after {time.time() - t0:.0f} s", flush=True)


# bench_common looks both up at call time, so measured() and its Sampler use these.
bc.other_gpu_jobs = other_gpu_jobs
bc.wait_for_idle_gpu = wait_for_idle_gpu


# ---------------------------------------------------------------------------------------------- grading
def _listify(targets):
    return list(targets) if isinstance(targets, (list, tuple)) else [targets]


def _objects_named(target, objects):
    """Scene object ids a mapping entry selects; PickPlaceTask.update_scene matches name or category."""
    return {k for k, cat in objects.items() if str(target) in (k, cat)}


def _resolve(group, mapping, objects):
    """Scene object ids a predicate's group stands for: through semantic_group_mapping, else itself."""
    if group is None:
        return set()
    return {k for t in _listify(mapping.get(group, group)) for k in _objects_named(t, objects)}


def _is_true(value):
    return value is None or value is True or str(value).lower() == "true"


def grade(tasks, objects):
    """Per written task: its predicates on the objects they resolve to, and the marker-into-cup check.

    The check reads the answer's intent: a predicate group missing from the mapping still resolves
    through the group's own name. Such a task would not load (PickPlaceTask indexes its groups by
    mapping key), which ``invalid`` records.
    """
    marker = next(k for k, cat in objects.items() if "marker" in cat)
    cup = next(k for k, cat in objects.items() if "cup" in cat)
    out = []
    for t in tasks[:NUM_TASKS]:
        mapping = t.get("semantic_group_mapping") or {}
        mapping = mapping if isinstance(mapping, dict) else {}
        preds = [p for key in ("goal_predicates_all", "goal_predicates_any") for p in (t.get(key) or [])
                 if isinstance(p, dict)]
        sig, hit = [], False
        for p in preds:
            g, o = _resolve(p.get("group"), mapping, objects), _resolve(p.get("other_group"), mapping, objects)
            names = lambda ids: "+".join(sorted(objects[i] for i in ids)) or "-"  # noqa: E731
            sig.append(f"{p.get('state')}({names(g)}{', ' + names(o) if p.get('other_group') else ''})"
                       + ("" if _is_true(p.get("value")) else "=false"))
            hit |= p.get("state") == "InsideAABB" and _is_true(p.get("value")) and g == {marker} and o == {cup}
        invalid = [f"group {x!r} is not in the mapping" for p in preds for x in (p.get("group"), p.get("other_group"))
                   if x is not None and x not in mapping]
        invalid += [f"{label}: {t!r} is no scene object" for label, targets in mapping.items()
                    for t in _listify(targets) if not _objects_named(t, objects)]
        # A label naming one object's kind but mapped to another (e.g. marker: [iter_1], the scissors).
        mislabeled = []
        for label, targets in mapping.items():
            for kind in ("marker", "cup", "scissors"):
                want = {k for k, cat in objects.items() if kind in cat}
                if kind in str(label).lower() and _resolve(label, mapping, objects) != want:
                    mislabeled.append(f"{label}: {targets}")
        out.append({"task_name": t.get("task_name"), "predicates": sig, "marker_in_cup": hit,
                    "invalid": invalid, "mislabeled": mislabeled})
    return out


def _task_files(scene, tasks):
    """The task YAMLs the stage writes for these tasks, as comparable strings."""
    base = roc.B7.get_base_task_template()
    return sorted(json.dumps(OmegaConf.to_container(OmegaConf.create(roc.B7.build_full_task_yaml(t, base, scene))),
                             sort_keys=True) for t in tasks[:NUM_TASKS])


def pipeline_task_files(scene):
    return sorted(json.dumps(OmegaConf.to_container(OmegaConf.load(p)), sort_keys=True)
                  for p in (bc.DATA / scene / "proposed_tasks").glob(f"{scene}_*.yaml"))


def fisher_two_sided(a, b, c, d):
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]]."""
    n1, n2, k = a + b, c + d, a + c
    p = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)  # noqa: E731
    p0 = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= p0 * (1 + 1e-9)))


# ---------------------------------------------------------------------------------------------- calls
def call(scene, seed):
    # The stage makes this call once in a fresh process, so the shim's repeat counter is empty and
    # the seed reaches Ollama unchanged; clearing it keeps that true for a call sent again here.
    bc.lb._REPEATS.clear()
    res, meta = bc.measured(lambda: roc.run_b7(scene, seed))
    meta["foreign_models"] = sorted({m[0] for m in meta["ollama_ps"] if m[0] != bc.lb.OLLAMA_MODEL})
    return res, meta, bool(meta["other_gpu_jobs"] or meta["foreign_models"])


def summarize(out, records):
    for r in records:  # graded again, so a change to grade() applies to calls already made
        r["graded"] = grade(r.get("tasks", []), r["objects"]) if r.get("objects") else []
    main_runs = {s: [r for r in records if r["case"]["scene"] == s and r["case"]["kind"] == "seed"
                     and not r["discarded"]] for s in SCENES}
    summary = {}
    for scene, runs in main_runs.items():
        runs.sort(key=lambda r: r["case"]["seed"])
        attempts = [a for r in runs for a in r["attempts"]]
        written = [g for r in runs for g in r.get("graded", [])]
        repeats = [r for r in records if r["case"]["scene"] == scene and r["case"]["kind"] == "repeat"
                   and not r["discarded"]]
        first = {r["case"]["seed"]: r.get("text") for r in runs}
        seed0 = next((r for r in runs if r["case"]["seed"] == 0), None)
        summary[scene] = {
            "objects": next((r["objects"] for r in runs if r.get("objects")), {}),
            "runs": len(runs),
            "discarded": sum(r["case"]["scene"] == scene and r["discarded"] for r in records),
            "errors": sum(bool(r.get("error")) for r in runs),
            "marker_in_cup_runs": sum(any(g["marker_in_cup"] for g in r.get("graded", [])) for r in runs),
            "marker_in_cup_seeds": [r["case"]["seed"] for r in runs if any(g["marker_in_cup"] for g in r.get("graded", []))],
            "marker_in_cup_runs_loadable": sum(any(g["marker_in_cup"] and not g["invalid"] for g in r.get("graded", []))
                                               for r in runs),
            "n_tasks_parsed": dict(sorted(Counter(r.get("n_tasks", 0) for r in runs).items())),
            "tasks_written": len(written),
            "mislabeled_tasks": sum(bool(g["mislabeled"]) for g in written),
            "invalid_tasks": sum(bool(g["invalid"]) for g in written),
            "task_kinds": Counter(" & ".join(g["predicates"]) for g in written).most_common(),
            "seed0_matches_pipeline": seed0 is not None and _task_files(scene, seed0.get("tasks", [])) == pipeline_task_files(scene),
            "repeats_identical": f"{sum(r.get('text') == first.get(r['case']['seed']) for r in repeats)}/{len(repeats)}",
            "max_tokens_hits": sum(a.get("done_reason") == "length" for a in attempts),
            "prompt_tokens": sorted({a.get("prompt_tokens") for a in attempts}),
            "gen_tokens_mean": round(sum(a.get("gen_tokens") or 0 for a in attempts) / max(1, len(attempts)), 1),
            "loaded_mib_vram_ctx": sorted({(m[1] // 2**20, m[2] // 2**20, m[3]) for r in runs for m in r["ollama_ps"]}),
        }
    a, c = (summary[s]["marker_in_cup_runs"] for s in SCENES)
    b, d = (summary[s]["runs"] - summary[s]["marker_in_cup_runs"] for s in SCENES)
    summary["fisher_p_marker_in_cup"] = fisher_two_sided(a, b, c, d)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")

    lines = ["| | " + " | ".join(SCENES) + " |", "|---|" + "---|" * len(SCENES)]
    row = lambda name, f: lines.append(f"| {name} | " + " | ".join(str(f(summary[s])) for s in SCENES) + " |")  # noqa: E731
    row("objects", lambda x: ", ".join(f"{k} {v}" for k, v in sorted(x["objects"].items())))
    row("runs kept (discarded as contended)", lambda x: f"{x['runs']} ({x['discarded']})")
    row("marker into cup (InsideAABB)", lambda x: f"{x['marker_in_cup_runs']}/{x['runs']}")
    row("... of which the YAML would load", lambda x: f"{x['marker_in_cup_runs_loadable']}/{x['runs']}")
    row("seeds with it", lambda x: x["marker_in_cup_seeds"])
    row("tasks parsed per run", lambda x: x["n_tasks_parsed"])
    row("written tasks: would not load / mislabeled", lambda x: f"{x['invalid_tasks']} / {x['mislabeled_tasks']} of {x['tasks_written']}")
    row("seed 0 = pipeline's YAMLs", lambda x: x["seed0_matches_pipeline"])
    row("repeated seed, same text", lambda x: x["repeats_identical"])
    row("MAX_TOKENS, prompt tokens, mean gen tokens",
        lambda x: f"{x['max_tokens_hits']}, {x['prompt_tokens']}, {x['gen_tokens_mean']}")
    row("model size / in VRAM (MiB) / context", lambda x: x["loaded_mib_vram_ctx"])
    lines.append(f"\nFisher exact (two-sided), marker into cup: p = {summary['fisher_p_marker_in_cup']:.3g}\n")
    for s in SCENES:
        lines.append(f"\n{s} written tasks:\n" + "\n".join(f"- {n} x {k}" for k, n in summary[s]["task_kinds"]))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--seeds", type=int, default=20, help="seeds 0..N-1 per scene")
    ap.add_argument("--repeat", default="0,1", help="seeds sent a second time, to check they reproduce")
    ap.add_argument("--data-root", default=str(bc.DATA), help="scene outputs (default: <repo>/Data)")
    ap.add_argument("--summarize-only", action="store_true", help="grade the calls already made; no VLM calls")
    args = ap.parse_args()
    bc.DATA = Path(args.data_root)
    out = Path(args.out_dir)
    calls = out / "calls.jsonl"
    records = [json.loads(line) for line in open(calls)] if calls.exists() else []
    if args.summarize_only:
        summarize(out, records)
        return
    bc.check_setup()
    assert bc.lb.OLLAMA_KEEP_ALIVE in ("0", "0s"), "set SIMFOUNDRY_OLLAMA_KEEP_ALIVE=0, as the pipeline does"
    for scene in SCENES:
        # The stage prefers assets/scenes/<scene>/; without it, it reads what run_b7 reads.
        assert not (bc.REPO / "assets/scenes" / scene).exists(), f"assets/scenes/{scene} would change B7's inputs"
    (out / "proposed_tasks").mkdir(parents=True, exist_ok=True)
    print(f"== {bc.lb.OLLAMA_MODEL} think={bc.lb.OLLAMA_THINK!r} keep_alive={bc.lb.OLLAMA_KEEP_ALIVE} "
          f"num_tasks={NUM_TASKS} temperature={roc.CFG.propose_scene_task.get('temperature', 0.7)}", flush=True)
    bc.warmup()

    def run(scene, seed, kind):
        done = {(r["case"]["scene"], r["case"]["seed"], r["case"]["kind"]) for r in records if not r["discarded"]}
        if (scene, seed, kind) in done:
            return
        for attempt in range(MAX_TRIES):
            res, meta, contended = call(scene, seed)
            case = {"id": f"B7_{scene}_seed{seed}{'_repeat' if kind == 'repeat' else ''}", "scene": scene,
                    "seed": seed, "kind": kind, "try": attempt}
            graded = grade(res.get("tasks", []), res["objects"]) if "objects" in res else []
            rec = {"case": case, "discarded": contended, **meta, **res, "graded": graded}
            bc.append_jsonl(calls, rec)
            records.append(json.loads(json.dumps(rec, default=str)))
            hit = any(g["marker_in_cup"] for g in graded)
            print(bc.summary_line(case["id"], meta, res.get("error") or
                                  f"{res.get('n_tasks')} parsed {[g['task_name'] for g in graded]}"
                                  f"{' MARKER->CUP' if hit else ''}{' (discarded)' if contended else ''}"), flush=True)
            if not contended:
                if kind == "seed":
                    for t in res.get("tasks", [])[:NUM_TASKS]:
                        name = re.sub(r"[^a-zA-Z0-9_-]", "_", t.get("task_name") or "task")
                        full = roc.B7.build_full_task_yaml(t, roc.B7.get_base_task_template(), scene)
                        OmegaConf.save(OmegaConf.create(full), out / "proposed_tasks" / f"{scene}_seed{seed}_{name}.yaml")
                return
        print(f"   gave up on {scene} seed {seed}: every try overlapped another job", flush=True)

    # Interleaved, so anything that drifts over the session falls on both scenes alike.
    for seed in range(args.seeds):
        for scene in SCENES:
            run(scene, seed, "seed")
    for seed in [int(s) for s in args.repeat.split(",") if s.strip()]:
        for scene in SCENES:
            run(scene, seed, "repeat")
    summarize(out, records)


if __name__ == "__main__":
    main()
