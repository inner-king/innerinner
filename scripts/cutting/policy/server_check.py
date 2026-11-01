# SPDX-License-Identifier: Apache-2.0
"""Sends a few fake observations to a running policy_server.py and prints the actions (wire and API check)."""
import sys
import time
from multiprocessing.connection import Client

import numpy as np


def pack(a):
    a = np.ascontiguousarray(a)
    return (a.dtype.str, a.shape, a.tobytes())


c = Client(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 6000), authkey=b"tofu")
c.send({"cmd": "reset"})
print("reset:", c.recv())
for i in range(3):
    t0 = time.perf_counter()
    c.send({"cmd": "act", "state": pack(np.zeros(16, np.float32)), "env_state": pack(np.zeros(7, np.float32)),
            "front": pack(np.zeros((224, 224, 3), np.uint8)), "wrist": pack(np.zeros((224, 224, 3), np.uint8))})
    a = c.recv()
    act = np.frombuffer(a[2], dtype=a[0]).reshape(a[1])
    print(i, act.shape, np.round(act, 3), f"{1e3 * (time.perf_counter() - t0):.1f} ms")
c.close()
