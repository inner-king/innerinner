# SPDX-License-Identifier: Apache-2.0
"""Dicing demonstrations (tofu_policy_episodes.py --mode dice: ep_*.npz + .json) to a local LeRobot image dataset.

Every variant reads the same episode files, so a new policy input needs no new demonstrations:
  base  observation.state (9: fingers, hand position and quaternion, base frame) + images front / wrist (224x224)
  A     the state plus the finished-cut count (the coupling's cuts begun less the one under way, per tick)
  C     variant A plus observation.environment_state (7: the block's centre, yaw and size, the evaluator's `blk_now`, constant
        within an episode; privileged: a perception module would have to supply it)
  B     the state now, 0.5 s and 1 s ago (27) + images front / wrist / wrist_past (the wrist frame 1 s ago): ACT in
        LeRobot takes one observation step, so the history goes in as inputs (policy_server.py --dice-variant B
        rebuilds the same from what it is sent)
Action at tick i = the end-effector pose commanded at tick i+1 (position and the first two rotation columns).
Run in the lerobot env:  python dice_to_lerobot.py --list accepted.txt --root /content/ds_dice --variant A
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

TASK = "dice the tofu on the board into a grid of cubes"  # policy_server.py DICE_TASK
STATE = ["finger0", "finger1", "ex", "ey", "ez", "eqx", "eqy", "eqz", "eqw"]
STATE_COLS = slice(7, 16)
ACTION = ["x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21"]
LAGS = (0, 5, 10)  # ticks at 10 Hz: now, 0.5 s and 1 s ago (variant B)
BLOCK = ["bx", "by", "bz", "byaw", "bsx", "bsy", "bsz"]


def features(variant, img):
    names = list(STATE)
    if variant in ("A", "C"):
        names = names + ["cuts_done"]
    elif variant == "B":
        names = [f"{n}_t-{lag}" for lag in LAGS for n in STATE]
    f = {"observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
         "action": {"dtype": "float32", "shape": (len(ACTION),), "names": ACTION}}
    if variant == "C":
        f["observation.environment_state"] = {"dtype": "float32", "shape": (len(BLOCK),), "names": BLOCK}
    cams = ("front", "wrist", "wrist_past") if variant == "B" else ("front", "wrist")
    for cam in cams:
        f[f"observation.images.{cam}"] = {"dtype": "video", "shape": (img, img, 3), "names": ["height", "width", "channels"]}
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, required=True, help="accepted npz paths, one per line (dice_select.py)")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--variant", choices=["base", "A", "B", "C"], default="base")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--vcodec", default="h264")
    ap.add_argument("--video-mb", type=int, default=2)
    ap.add_argument("--skip-ticks", type=int, default=5, help="each episode's first ticks (DLSS ghosts after a reset)")
    args = ap.parse_args()
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    paths = [Path(p) for p in args.list.read_text().split()]
    root = args.root / f"dice_{args.variant}"
    if root.exists():
        shutil.rmtree(root)
    img = int(np.load(paths[0])["images_front"].shape[1])
    enc = {"rgb_encoder": RGBEncoderConfig(vcodec=args.vcodec, pix_fmt="yuv420p", preset="veryfast", crf=20)}
    ds = LeRobotDataset.create(repo_id=f"local/dice_{args.variant}", fps=args.fps, root=root, robot_type="franka",
                               features=features(args.variant, img), use_videos=True, image_writer_threads=8,
                               video_files_size_in_mb=args.video_mb, **enc)
    for path in paths:
        z = np.load(path)
        st, cmd = z["state"][:, STATE_COLS].astype(np.float32), z["cmd"]  # read each array once
        front, wrist, cuts = z["images_front"], z["images_wrist"], z["tick_cuts_done"].astype(np.float32)
        act = np.concatenate([cmd[1:], cmd[-1:]], axis=0).astype(np.float32)
        for i in range(args.skip_ticks, len(cmd)):
            if args.variant in ("A", "C"):
                s_i = np.append(st[i], cuts[i]).astype(np.float32)
            elif args.variant == "B":
                s_i = np.concatenate([st[max(args.skip_ticks, i - lag)] for lag in LAGS]).astype(np.float32)
            else:
                s_i = st[i]
            fr = {"observation.state": s_i, "action": act[i], "task": TASK,
                  "observation.images.front": front[i], "observation.images.wrist": wrist[i]}
            if args.variant == "B":
                fr["observation.images.wrist_past"] = wrist[max(args.skip_ticks, i - LAGS[-1])]
            if args.variant == "C":
                fr["observation.environment_state"] = z["block"].astype(np.float32)
            ds.add_frame(fr)
        ds.save_episode()
    if hasattr(ds, "finalize"):
        ds.finalize()
    print(args.variant, root, ds.meta.total_episodes, ds.meta.total_frames)


if __name__ == "__main__":
    main()
