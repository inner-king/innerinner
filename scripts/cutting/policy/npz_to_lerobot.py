# SPDX-License-Identifier: Apache-2.0
"""Episodes of tofu_policy_episodes.py (ep_*.npz + .json) to two local LeRobot datasets.

  <root>/tofu_state  observation.state (9: fingers, hand position and quaternion, base frame) +
                     observation.environment_state (7: the block's centre, yaw, size; privileged), action (9)
  <root>/tofu_image  observation.state (9) + observation.images.front / .wrist (224x224 RGB), action (9)

Action at tick i = the end-effector pose commanded at tick i+1 (base frame: position and the first two
rotation columns); the last tick repeats its own command. Only episodes the solver judged a success go in.

--action-height log-support (the support-height target, see simfoundry/cutting/support_height.py): the action's z
becomes u = ln(h + eps), h being the lowest point of the knife edge above the block top at the commanded pose, and the
state gets the block top s appended. Both are the evaluation client's definitions, so a policy trained here answers in
the same heights. The default (absolute) is unchanged and imports nothing from simfoundry.

Run in the lerobot env:  python npz_to_lerobot.py --src DIR [DIR ...] --root /content/ds
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

TASK = "cut the tofu on the board into two pieces"
# The policy sees the hand, not the arm's joints: a task-space state carries over to another arm, and the
# Franka's redundant posture says nothing about the cut. Episode state columns 7.. are these.
STATE = ["finger0", "finger1", "ex", "ey", "ez", "eqx", "eqy", "eqz", "eqw"]
STATE_COLS = slice(7, 16)
ACTION = ["x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21"]
ACTION_LOG = ["u" if name == "z" else name for name in ACTION]
BLOCK = ["cx", "cy", "cz", "yaw", "sx", "sy", "sz"]
# The knife of an episode whose JSON has none is fitted to the folder's first episodes that the setup does not explain.
# 9 is demonstrations' default (support_height_roundtrip.py --fit-episodes); the dataset meta records what was fitted.
FIT_EPISODES = 9


def quat_xyzw_to_matrix(q):
    """(N, 4) xyzw quaternions to (N, 3, 3) rotation matrices (no scipy in the lerobot env)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1)], axis=1)


def _simfoundry():
    """The simfoundry modules log-support uses. Imported here, not at the top, so the default path never needs them."""
    repo = str(Path(__file__).resolve().parents[3])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from simfoundry.cutting import demonstrations, support_height

    return demonstrations, support_height


def features(kind, img, drop=False, support=False):
    st = STATE[2:] if drop else STATE
    ac = ACTION_LOG if support else ACTION
    if drop:
        ac = ac[:8]
    if support:
        st = st + ["support"]
    f = {"observation.state": {"dtype": "float32", "shape": (len(st),), "names": st},
         "action": {"dtype": "float32", "shape": (len(ac),), "names": ac}}
    if kind == "state":
        f["observation.environment_state"] = {"dtype": "float32", "shape": (7,), "names": BLOCK}
    else:
        for cam in ("front", "wrist"):
            f[f"observation.images.{cam}"] = {"dtype": "video", "shape": (img, img, 3),
                                              "names": ["height", "width", "channels"]}
    return f


def log_support_episode(path, eps, fit_episodes, cache):
    """The log-support action z and the block top for one demonstration. Nothing is written.

    Args:
        path: an ``ep_XXXXX.npz`` of tofu_policy_episodes.py; its folder holds ``setup.json``.
        eps: the epsilon of u = ln(h + eps) [m].
        fit_episodes: how many episodes a folder's fitted knife uses.
        cache: folder -> fit dict, shared by the episodes of one run (demonstrations.choose_knife fills it).

    Returns:
        A dict. ``u`` (T,) float64 is the action's z slot, one tick ahead: u[i] is the height of command i+1, and the
        last tick's is its own command. ``s`` [m] is the block top; ``knife_source`` is "episode", "setup" or "fitted";
        ``resid_rms_mm`` and ``resid_max_mm`` measure the knife against the recorded edge heights; ``h_mm`` holds the
        min, median and max of h over the ticks; ``ticks`` is T; ``path`` is the npz as given; ``fit`` is the folder's
        fit dict when the knife was fitted, else None.

    Raises:
        SystemExit: if a tick's edge is at or below the support by eps (the episode is named), or the knife has to be
            fitted in an environment without scipy.
    """
    demonstrations, support_height = _simfoundry()
    name = Path(path).name
    demo = demonstrations.load_demo(path)
    try:
        knife, source, fit = demonstrations.choose_knife(demo, fit_episodes, cache)
    except ImportError as err:  # fit_knife_in_eef imports scipy; the setup and the episode's own knife do not need it
        if (err.name or "").split(".")[0] == "scipy":
            raise SystemExit(
                f"{name}: this episode's knife transform has to be fitted (needs scipy): convert in an environment with "
                "scipy, or use demonstrations whose ep_*.json carry knife_in_eef"
            ) from err
        raise
    # The command of the next tick is the pose the hand is asked to reach; the last tick repeats its own.
    commanded = demo["E_cmd"]
    nxt = np.concatenate([commanded[1:], commanded[-1:]], axis=0)
    h = support_height.support_height(nxt, knife, demo["s"])
    if np.any(h + eps <= 0.0):
        raise SystemExit(f"{name}: {int(np.count_nonzero(h + eps <= 0.0))} tick(s) have h + eps <= 0 "
                         f"(smallest h {float(h.min()):.6f} m, eps {eps:g} m): the knife edge is at or below the block "
                         "top by eps there, so no log height exists")
    u = support_height.log_height(h, eps)
    resid = demonstrations.edge_residual_mm(demo, knife)
    return {
        "path": str(path),
        "name": name,
        "u": u,
        "s": float(demo["s"]),
        "knife_source": source,
        "resid_rms_mm": float(np.sqrt(np.mean(resid**2))),
        "resid_max_mm": float(np.max(np.abs(resid))),
        "h_mm": {"min": 1e3 * float(h.min()), "median": 1e3 * float(np.median(h)), "max": 1e3 * float(h.max())},
        "ticks": int(len(u)),
        "fit": fit,
    }


def support_meta(results, cache, eps):
    """The meta file of a log-support dataset: what the targets were, and which knife and fit each episode used."""
    _, support_height = _simfoundry()
    return {
        "eps_m": float(eps),
        "action_height": "log-support",
        "edge_samples": int(support_height.EDGE_SAMPLES),
        "episodes": [{"npz": r["path"], "knife_source": r["knife_source"], "resid_rms_mm": r["resid_rms_mm"],
                      "resid_max_mm": r["resid_max_mm"], "s_m": r["s"], "h_mm": r["h_mm"], "ticks": r["ticks"]}
                     for r in results],
        "fits": [{"folder": str(folder), "episodes": fit["episodes"], "fit_rms_mm": fit["fit_rms_mm"]}
                 for folder, fit in cache.items()],
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, nargs="*", default=[])
    ap.add_argument("--list", type=Path, default=None, help="accepted.txt from select_demos.py: npz paths, one per line")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--kinds", nargs="+", default=["state", "image"])
    ap.add_argument("--all", action="store_true", help="keep failed episodes too")
    ap.add_argument("--vcodec", default=None, help="e.g. h264 (fast to encode); default: LeRobot's AV1")
    ap.add_argument("--action", choices=["absolute", "delta"], default="absolute",
                    help="delta: change from the hand's pose now to the next command (position in the base frame, "
                         "turn in the hand's frame), so the few millimetres that decide the cut are not a sliver of "
                         "a range spanning the whole reach")
    ap.add_argument("--action-height", choices=["absolute", "log-support"], default="absolute",
                    help="log-support: the action's z is u = ln(h + eps) with h the knife edge's lowest point above the "
                         "block top, and the state gets the block top s appended (the evaluation client's definition: "
                         "policy_server takes the last n state entries, so n = 10 reads the hand and s)")
    ap.add_argument("--eps-mm", type=float, default=1.0,
                    help="eps of the log height [mm]; log-support needs it to equal simfoundry's EPS_M (ignored otherwise)")
    ap.add_argument("--video-mb", type=int, default=2, help="video file size limit [MB]")
    ap.add_argument("--drop-near-constant", action="store_true",
                    help="leave out the near-constant columns (action r21: std 0.0019; state fingers: std 0.0008), which "
                         "a normalised L1 loss would otherwise weigh up; the executor refills r21 as 0 before Gram-Schmidt")
    ap.add_argument("--skip-ticks", type=int, default=0,
                    help="drop each episode's first ticks (image kind: DLSS ghosts of the last block after a reset)")
    return ap.parse_args(argv)


def check_args(args):
    """Refuses the options that cannot make a log-support dataset; simfoundry is imported only for log-support.

    Raises:
        SystemExit: on delta with log-support, or an eps that differs from the evaluation client's EPS_M.
    """
    if args.action_height != "log-support":
        return
    if args.action == "delta":
        raise SystemExit("--action-height log-support cannot be combined with --action delta: the delta's z is a change "
                         "in height, not a log height")
    _, support_height = _simfoundry()
    if abs(args.eps_mm - 1e3 * support_height.EPS_M) > 1e-9:
        raise SystemExit(f"--eps-mm {args.eps_mm:g} differs from simfoundry.cutting.support_height.EPS_M "
                         f"({1e3 * support_height.EPS_M:g} mm): the evaluation client uses EPS_M, so a policy trained "
                         "with another eps would answer with the wrong heights. To change eps, change EPS_M in a "
                         "separate decision.")


def main(argv=None):
    args = parse_args(argv)
    check_args(args)
    support = args.action_height == "log-support"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    eps = []
    if args.list is not None:
        for line in args.list.read_text().split():
            j = Path(line).with_suffix(".json")
            eps.append((Path(line), json.loads(j.read_text())))
    for d in args.src:
        for j in sorted(d.glob("ep_*.json")):
            info = json.loads(j.read_text())
            if args.all or info["outcome"]["success"]:
                eps.append((j.with_suffix(".npz"), info))
    print(f"{len(eps)} episodes")
    cache = {}
    results = []
    if support:
        _, support_height = _simfoundry()
        eps_m = 1e-3 * args.eps_mm
        # All episodes are converted before any dataset is written, so a bad episode stops the run with nothing half-made.
        results = [log_support_episode(path, eps_m, FIT_EPISODES, cache) for path, _ in eps]
    for kind in args.kinds:
        root = args.root / f"tofu_{kind}"
        if root.exists():
            shutil.rmtree(root)
        e0 = np.load(eps[0][0])
        img = int(e0["images_front"].shape[1])
        enc = {}
        if args.vcodec:
            from lerobot.configs.video import RGBEncoderConfig

            enc["rgb_encoder"] = RGBEncoderConfig(vcodec=args.vcodec, pix_fmt="yuv420p", preset="veryfast", crf=20)
        # Small video files: v3 appends each episode to the current chunk file, which costs more as it grows.
        ds = LeRobotDataset.create(repo_id=f"local/tofu_{kind}", fps=args.fps, root=root, robot_type="franka",
                                   features=features(kind, img, args.drop_near_constant, support), use_videos=kind == "image",
                                   image_writer_threads=8,
                                   video_files_size_in_mb=args.video_mb, **enc)
        for k, (path, info) in enumerate(eps):
            z = np.load(path)
            e = {k2: z[k2] for k2 in ("cmd", "state", "block") + (("images_front", "images_wrist") if kind == "image" else ())}
            cmd = e["cmd"]  # (an NpzFile decompresses an array on every access: read each once)
            act = np.concatenate([cmd[1:], cmd[-1:]], axis=0).astype(np.float32)
            if args.action == "delta":
                st = e["state"]
                Rh = quat_xyzw_to_matrix(st[:, 12:16])
                b1, b2 = act[:, 3:6], act[:, 6:9]
                Rc = np.stack([b1, b2, np.cross(b1, b2)], axis=2)
                dR = np.einsum("nji,njk->nik", Rh, Rc)  # R_hand^T R_cmd
                act = np.concatenate([act[:, :3] - st[:, 9:12], dR[:, :, 0], dR[:, :, 1]], axis=1).astype(np.float32)
            if support:
                act[:, 2] = results[k]["u"].astype(np.float32)
            for i in range(args.skip_ticks if kind == "image" else 0, len(cmd)):
                st_i = e["state"][i, STATE_COLS].astype(np.float32)
                a_i = act[i]
                if args.drop_near_constant:
                    st_i, a_i = st_i[2:], a_i[:8]
                if support:
                    st_i = support_height.state_with_support(st_i, results[k]["s"])
                fr = {"observation.state": st_i, "action": a_i, "task": TASK}
                if kind == "state":
                    fr["observation.environment_state"] = e["block"].astype(np.float32)
                else:
                    fr["observation.images.front"] = e["images_front"][i]
                    fr["observation.images.wrist"] = e["images_wrist"][i]
                ds.add_frame(fr)
            ds.save_episode()
        if hasattr(ds, "finalize"):
            ds.finalize()
        if support:
            (root / "meta").mkdir(parents=True, exist_ok=True)
            meta = support_meta(results, cache, eps_m)
            (root / "meta" / "support_height.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(kind, root, ds.meta.total_episodes, ds.meta.total_frames)


if __name__ == "__main__":
    main()
