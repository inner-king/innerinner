# SPDX-License-Identifier: Apache-2.0
"""Dicing server smoke test: starts policy_server.py on the CPU with a checkpoint, sends a reset and 12 ticks of made-up
observations with a finished-cut count, and checks that every answer is nine finite numbers. Says nothing about a
policy; it checks the wire format and the --dice-variant inputs (A: cut count, B: history) end to end.

Usage: dice_server_smoke.py <checkpoint dir> <A|B> [--port 6099]
"""
import argparse
import subprocess
import sys
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np


def pack(a):
    a = np.ascontiguousarray(a)
    return (a.dtype.str, a.shape, a.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("variant", choices=["A", "B"])
    ap.add_argument("--port", type=int, default=6099)
    args = ap.parse_args()
    server = Path(__file__).with_name("policy_server.py")
    srv = subprocess.Popen([sys.executable, str(server), "--ckpt", args.ckpt, "--port", str(args.port), "--device", "cpu",
                            "--dice-variant", args.variant], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
            c.send({"cmd": "act", "state": pack(np.random.rand(16).astype(np.float32)), "env_state": pack(np.zeros(7, np.float32)),
                    "front": pack(img), "wrist": pack(img), "cuts_done": k // 3})
            a = c.recv()
            out.append(np.frombuffer(a[2], dtype=a[0]).reshape(a[1]))
        c.close()
    finally:
        srv.terminate()
    ok = all(x.shape == (9,) and np.isfinite(x).all() for x in out)
    print(f"variant {args.variant}: {'OK' if ok else 'FAIL'}, {len(out)} actions, first {np.round(out[0][:3], 3)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
