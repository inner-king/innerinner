# SPDX-License-Identifier: Apache-2.0
"""Episodes of tofu_policy_episodes.py (ep_*.npz + .json) to two local LeRobot datasets.

  <root>/tofu_state  observation.state (9: fingers, hand position and quaternion, base frame) +
                     observation.environment_state (7: the block's centre, yaw, size; privileged), action (9)
  <root>/tofu_image  observation.state (9) + observation.images.front / .wrist (224x224 RGB), action (9)

Action at tick i = the end-effector pose commanded at tick i+1 (base frame: position and the first two
rotation columns); the last tick repeats its own command. Only episodes the solver judged a success go in.
Run in the lerobot env:  python npz_to_lerobot.py --src DIR [DIR ...] --root /content/ds
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

TASK = "cut the tofu on the board into two pieces"
# The policy sees the hand, not the arm's joints: a task-space state carries over to another arm, and the
# Franka's redundant posture says nothing about the cut. Episode state columns 7.. are these.
STATE = ["finger0", "finger1", "ex", "ey", "ez", "eqx", "eqy", "eqz", "eqw"]
STATE_COLS = slice(7, 16)
ACTION = ["x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21"]
BLOCK = ["cx", "cy", "cz", "yaw", "sx", "sy", "sz"]


def quat_xyzw_to_matrix(q):
    """(N, 4) xyzw quaternions to (N, 3, 3) rotation matrices (no scipy in the lerobot env)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1)], axis=1)


def features(kind, img, drop=False):
    st, ac = (STATE[2:], ACTION[:8]) if drop else (STATE, ACTION)
    f = {"observation.state": {"dtype": "float32", "shape": (len(st),), "names": st},
         "action": {"dtype": "float32", "shape": (len(ac),), "names": ac}}
    if kind == "state":
        f["observation.environment_state"] = {"dtype": "float32", "shape": (7,), "names": BLOCK}
    else:
        for cam in ("front", "wrist"):
            f[f"observation.images.{cam}"] = {"dtype": "video", "shape": (img, img, 3),
                                              "names": ["height", "width", "channels"]}
    return f


def main():
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
    ap.add_argument("--video-mb", type=int, default=2, help="video file size limit [MB]")
    ap.add_argument("--drop-near-constant", action="store_true",
                    help="leave out the near-constant columns (action r21: std 0.0019; state fingers: std 0.0008), which "
                         "a normalised L1 loss would otherwise weigh up; the executor refills r21 as 0 before Gram-Schmidt")
    ap.add_argument("--skip-ticks", type=int, default=0,
                    help="drop each episode's first ticks (image kind: DLSS ghosts of the last block after a reset)")
    args = ap.parse_args()
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
                                   features=features(kind, img, args.drop_near_constant), use_videos=kind == "image",
                                   image_writer_threads=8,
                                   video_files_size_in_mb=args.video_mb, **enc)
        for path, info in eps:
            z = np.load(path)
            e = {k: z[k] for k in ("cmd", "state", "block") + (("images_front", "images_wrist") if kind == "image" else ())}
            cmd = e["cmd"]  # (an NpzFile decompresses an array on every access: read each once)
            act = np.concatenate([cmd[1:], cmd[-1:]], axis=0).astype(np.float32)
            if args.action == "delta":
                st = e["state"]
                Rh = quat_xyzw_to_matrix(st[:, 12:16])
                b1, b2 = act[:, 3:6], act[:, 6:9]
                Rc = np.stack([b1, b2, np.cross(b1, b2)], axis=2)
                dR = np.einsum("nji,njk->nik", Rh, Rc)  # R_hand^T R_cmd
                act = np.concatenate([act[:, :3] - st[:, 9:12], dR[:, :, 0], dR[:, :, 1]], axis=1).astype(np.float32)
            for i in range(args.skip_ticks if kind == "image" else 0, len(cmd)):
                st_i = e["state"][i, STATE_COLS].astype(np.float32)
                a_i = act[i]
                if args.drop_near_constant:
                    st_i, a_i = st_i[2:], a_i[:8]
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
        print(kind, root, ds.meta.total_episodes, ds.meta.total_frames)


if __name__ == "__main__":
    main()
