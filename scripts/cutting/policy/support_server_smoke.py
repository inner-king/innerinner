# SPDX-License-Identifier: Apache-2.0
"""Support-height policy smoke test: starts policy_server.py on the CPU with a checkpoint, sends a reset and 12 ticks of
made-up observations, and checks that every answer is nine finite numbers and that exp(a[2]) - EPS_M, the log-support
height, is finite. Says nothing about a policy; it checks the wire format of a log-support state (16 hand and state
entries with the block top s appended, 17 in all; the server reads the last n = 10 of them) and the reading of the z slot.

Usage: support_server_smoke.py <checkpoint dir> <state|image> [--port 6098] [--s 0.0110437]
"""
import argparse
import subprocess
import sys
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

EPS_M = 1.0e-3  # simfoundry.cutting.support_height.EPS_M; written out so the lerobot env need not import simfoundry


def pack(a):
    a = np.ascontiguousarray(a)
    return (a.dtype.str, a.shape, a.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("kind", choices=["state", "image"])
    ap.add_argument("--port", type=int, default=6098)
    ap.add_argument("--s", type=float, default=0.0110437, help="the block top sent as the 17th state entry [m]")
    args = ap.parse_args()
    server = Path(__file__).with_name("policy_server.py")
    srv = subprocess.Popen([sys.executable, str(server), "--ckpt", args.ckpt, "--port", str(args.port), "--device", "cpu"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in srv.stdout:
            if line.startswith("ready"):
                break
        else:
            raise SystemExit("the server exited before it was ready")
        c = Client(("127.0.0.1", args.port), authkey=b"tofu")
        c.send({"cmd": "reset"})
        c.recv()
        out = []
        for k in range(12):
            img = np.full((224, 224, 3), 20 * k, np.uint8)
            state = np.append(np.random.rand(16), args.s).astype(np.float32)
            c.send({"cmd": "act", "state": pack(state), "env_state": pack(np.zeros(7, np.float32)),
                    "front": pack(img), "wrist": pack(img)})
            a = c.recv()
            out.append(np.frombuffer(a[2], dtype=a[0]).reshape(a[1]))
        c.close()
    finally:
        srv.terminate()
    shapes_ok = all(x.shape == (9,) for x in out)
    finite = all(np.isfinite(x).all() for x in out)
    heights = np.array([np.exp(x[2]) - EPS_M for x in out])
    heights_ok = bool(np.isfinite(heights).all())
    ok = shapes_ok and finite and heights_ok
    print(f"{args.kind}: {'OK' if ok else 'FAIL'}, {len(out)} actions, shapes {'(9,)' if shapes_ok else 'BAD'}, "
          f"finite {finite}, heights [m] min {heights.min():.4f} max {heights.max():.4f}, "
          f"first action {np.round(out[0], 3)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
