# SPDX-License-Identifier: Apache-2.0
"""Send a few DROID-shaped observations to an openpi policy server and report the action chunk and latency.

Used by run_table_g1.sh to warm a server up (its first call compiles, about 30 s on the G4) before any
episode waits on it, and as a check that the server returns absolute joint targets: the first action should
sit near the joint state that was sent. The observation keys are the ones simfoundry/policies/openpi.py
sends; the joint state is the DROID scenes' saved arm pose.

    python repro/policy_eval/ping_policy.py --port 8001
"""
import argparse
import time

import numpy as np
from openpi_client import websocket_client_policy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--prompt", default="pick up the marker, and put the marker in the cup")
    args = ap.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    rng = np.random.default_rng(0)
    joints = np.array([0.0, -0.823, 0.0, -2.371, 0.0, 1.51, 0.0], dtype=np.float32)
    obs = {
        "observation/exterior_image_1_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/exterior_image_2_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/joint_position": joints,
        "observation/gripper_position": np.array([0.0], dtype=np.float32),
        "prompt": args.prompt,
    }
    for i in range(args.n):
        t0 = time.time()
        actions = np.asarray(client.infer(obs)["actions"])
        dt = time.time() - t0
        print(f"[ping :{args.port}] call {i}: {dt * 1000:.0f} ms, chunk {actions.shape}, "
              f"max |a0 - q| = {np.abs(actions[0, :7] - joints).max():.3f} rad", flush=True)


if __name__ == "__main__":
    main()
