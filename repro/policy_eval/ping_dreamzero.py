# SPDX-License-Identifier: Apache-2.0
"""Warm DreamZero's server up through the eval stage's own client, and report each call's latency.

The server compiles on its first calls (torch.compile, minutes per call, once per new context length), so
run_table_g1.sh makes them here rather than inside an episode. An "episode" is reset() and then --n infer() calls
with the observation the eval stage sends (its keys, 1280x720 cameras, the controller's reset pose), as
simfoundry/policies/dreamzero.py packs it. From the second episode on, the first call also carries the server's
session change, on which it decodes and writes the previous session's predicted video. The first action should
sit near the joint state sent: the server returns absolute joint targets.

    python repro/policy_eval/ping_dreamzero.py --port 5000 --n 8 --episodes 2 [--json latency.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The eval stage's reset pose (repro/README.md 18.2).
RESET_Q = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.0], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--n", type=int, default=8, help="calls per episode")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--prompt", default="pick up the marker, and put the marker in the cup")
    ap.add_argument("--json", default=None, help="write the per-call latencies here")
    args = ap.parse_args()
    sys.path.insert(0, REPO)
    from simfoundry.policies.dreamzero import DreamZeroClient

    client = DreamZeroClient(host=args.host, port=args.port, open_loop_horizon=8)
    rng = np.random.default_rng(0)
    obs = {
        "exterior_image_1_left": rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8),
        "exterior_image_2_left": rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8),
        "wrist_image_left": rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8),
        "joint_position": RESET_Q,
        "gripper_position": np.array([0.0], dtype=np.float32),
    }
    calls = []
    for ep in range(args.episodes):
        client.reset()
        for i in range(args.n):
            t0 = time.time()
            actions = client.infer(obs, args.prompt)["action"]
            dt = time.time() - t0
            calls.append({"episode": ep, "call": i, "s": dt})
            print(f"[ping_dreamzero :{args.port}] episode {ep} call {i}: {dt:.1f} s, chunk {actions.shape}, "
                  f"max |a0 - q| = {np.abs(actions[0, :7] - RESET_Q).max():.3f} rad, "
                  f"gripper {actions[0, 7]:.2f}", flush=True)
    last = [c["s"] for c in calls if c["episode"] == args.episodes - 1 and c["call"] > 0]
    if last:
        print(f"[ping_dreamzero :{args.port}] last episode, calls 1..{args.n - 1}: median {np.median(last):.2f} s, "
              f"max {max(last):.2f} s", flush=True)
    if args.json:
        json.dump(calls, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
