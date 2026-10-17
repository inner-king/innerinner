# SPDX-License-Identifier: Apache-2.0
"""How much the first rendered frames of two runs on the same starts differ (repro/README.md, 18.4).

1_eval_policy_og_scene.py records each episode's video from the observation it sends before the first action,
so two runs on one init-state file open every episode on the same scene. What can differ there is the
renderer: the GPU (18.4's repeats ran on a G4, the runs before them on L4s), the path tracer's sampling noise
and the mp4 encoding. For each pair of runs this prints the mean absolute pixel difference (0-255) of
episode k's first frame, per camera (the video concatenates the two external cameras and the wrist camera),
as the median over the episodes both runs recorded. Two runs on the same GPU type give the noise floor.

    python repro/policy_eval/first_frames.py \
        out/marker_in_cup/s15_eval_pi05_main/*/*/results_* out/marker_in_cup/s15_eval_pi05_prompt_repo/*/*/results_* \
        out/marker_in_cup/s15_eval_pi05_main/*/*/results_* out/marker_in_cup/s15_eval_pi05_rep2/*/*/results_*
"""
import argparse
import glob
import os

import cv2
import numpy as np

CAMERAS = ("ext_0", "ext_1", "wrist")


def first_frame(run, k):
    paths = glob.glob(os.path.join(run, "videos", f"episode_{k:03d}_*.mp4"))
    if not paths:
        return None
    cap = cv2.VideoCapture(paths[0])
    ok, frame = cap.read()
    cap.release()
    return frame.astype(np.float32) if ok else None


def run_name(run):
    """s15_eval_<policy>_<tag> of .../s15_eval_<policy>_<tag>/<policy>/<checkpoint>/results_*."""
    return os.path.normpath(run).split(os.sep)[-4]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="results_* directories, taken in pairs")
    ap.add_argument("--episodes", type=int, default=25)
    args = ap.parse_args()
    if len(args.runs) % 2:
        raise SystemExit("runs come in pairs")
    for a, b in zip(args.runs[::2], args.runs[1::2]):
        diffs = []
        for k in range(args.episodes):
            fa, fb = first_frame(a, k), first_frame(b, k)
            if fa is None or fb is None or fa.shape != fb.shape:
                continue
            w = fa.shape[1] // len(CAMERAS)
            diffs.append([np.abs(fa[:, i * w:(i + 1) * w] - fb[:, i * w:(i + 1) * w]).mean() for i in range(len(CAMERAS))])
        if not diffs:
            print(f"{run_name(a)} vs {run_name(b)}: no episode recorded by both")
            continue
        d = np.array(diffs)
        print(f"{run_name(a)} vs {run_name(b)}: {len(d)} episodes, median MAE "
              + ", ".join(f"{c} {np.median(d[:, i]):.2f}" for i, c in enumerate(CAMERAS)) + f"; max {d.max():.2f}")


if __name__ == "__main__":
    main()
