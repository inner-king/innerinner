# SPDX-License-Identifier: Apache-2.0
"""What the replay scripts record around each VLM call, without touching the call itself.

Every Ollama /api/chat attempt the shim makes is kept (tokens, durations, done_reason, the
thinking text), and device memory is sampled while the call runs. Importing this module puts the
repository root first on sys.path, so the shim under test is this checkout's.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pynvml  # noqa: E402
import requests  # noqa: E402

from simfoundry.models import local_backends as lb  # noqa: E402

# Where scenes are read from; the scripts' --data-root replaces it.
DATA = REPO / "Data"

ATTEMPTS = []
_post = requests.post


def _recording_post(url, *args, **kwargs):
    resp = _post(url, *args, **kwargs)
    if url.endswith("/api/chat"):
        body = kwargs.get("json") or {}
        try:
            j = resp.json()
            msg = j.get("message", {})
            ATTEMPTS.append({
                "done_reason": j.get("done_reason"),
                "prompt_tokens": j.get("prompt_eval_count"), "gen_tokens": j.get("eval_count"),
                "load_s": (j.get("load_duration") or 0) / 1e9, "prompt_s": (j.get("prompt_eval_duration") or 0) / 1e9,
                "gen_s": (j.get("eval_duration") or 0) / 1e9, "total_s": (j.get("total_duration") or 0) / 1e9,
                "thinking": msg.get("thinking") or "", "content": msg.get("content") or "",
                "think_sent": body.get("think", "unset"),
                "temperature": body.get("options", {}).get("temperature"), "seed": body.get("options", {}).get("seed"),
            })
        except Exception as exc:  # recording must never change the call
            ATTEMPTS.append({"record_error": repr(exc)})
    return resp


requests.post = _recording_post


def check_setup():
    """Refuse to run against the wrong shim or with the response cache on."""
    assert os.environ.get("SIMFOUNDRY_LOCAL_BACKEND") == "1", "set SIMFOUNDRY_LOCAL_BACKEND=1"
    # The cache key names the Gemini model a stage asked for, not the Ollama model that answers,
    # so a cache would replay another model's answers.
    assert not os.environ.get("CACHE_MODE") and not os.environ.get("TEST_MODE"), "run with the response cache off"
    assert Path(lb.__file__).resolve().is_relative_to(REPO), lb.__file__


pynvml.nvmlInit()
_GPU = pynvml.nvmlDeviceGetHandleByIndex(0)


def other_gpu_jobs():
    """PIDs computing on the GPU other than Ollama's runners."""
    pids = []
    for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(_GPU):
        try:
            cmd = Path(f"/proc/{proc.pid}/cmdline").read_bytes().decode(errors="replace")
        except OSError:
            cmd = ""
        if "ollama" not in cmd:
            pids.append(proc.pid)
    return pids


def wait_for_idle_gpu():
    """Another job on the card skews latency and memory, and can make Ollama load the model with
    smaller buffers, which changes the numerics enough to flip borderline greedy answers."""
    while other_gpu_jobs():
        time.sleep(10)


def used_mib():
    # Includes the driver's reserved memory (~0.4 GB here) that nvidia-smi leaves out; compare
    # peak - base, never the absolute value.
    return pynvml.nvmlDeviceGetMemoryInfo(_GPU).used / 2**20


class Sampler:
    """Peak device memory and the models Ollama reports loaded while a call runs."""

    def __init__(self):
        self.peak = 0.0
        self.ps = set()
        self.others = set()
        self._stop = threading.Event()

    def __enter__(self):
        self.base = used_mib()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        last_ps = 0.0
        while not self._stop.is_set():
            self.peak = max(self.peak, used_mib())
            if time.time() - last_ps > 1.0:
                last_ps = time.time()
                self.others.update(other_gpu_jobs())
                try:
                    for m in requests.get(f"{lb.OLLAMA_URL}/api/ps", timeout=2).json().get("models", []):
                        self.ps.add((m.get("name"), m.get("size"), m.get("size_vram"), m.get("context_length")))
                except Exception:
                    pass
            time.sleep(0.05)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def warmup():
    """Load and unload the model once, so the first measured call is not a cold disk read."""
    body = {"model": lb.OLLAMA_MODEL, "stream": False, "keep_alive": "0",
            "messages": [{"role": "user", "content": "ok"}], "options": {"num_predict": 1}}
    if lb.OLLAMA_THINK not in (None, ""):
        body["think"] = lb.OLLAMA_THINK.strip().lower() in {"1", "true", "yes", "on"}
    _post(f"{lb.OLLAMA_URL}/api/chat", json=body, timeout=600)
    lb._wait_until(lambda: not requests.get(f"{lb.OLLAMA_URL}/api/ps", timeout=10).json().get("models"), timeout_s=60)


def measured(fn):
    """Run ``fn()`` and return (its result dict, a record of the attempts, latency and memory)."""
    wait_for_idle_gpu()
    ATTEMPTS.clear()
    t0 = time.time()
    with Sampler() as smp:
        try:
            res = fn()
        except Exception as exc:
            import traceback
            res = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    return res, {
        "model": lb.OLLAMA_MODEL, "think_env": lb.OLLAMA_THINK, "wall_s": round(time.time() - t0, 2),
        "vram_base_mib": round(smp.base), "vram_peak_mib": round(smp.peak), "ollama_ps": sorted(smp.ps),
        "other_gpu_jobs": sorted(smp.others),
        "attempts": list(ATTEMPTS),
    }


def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def summary_line(case_id, meta, tail):
    a = meta["attempts"]
    return (f"{case_id:34s} {meta['wall_s']:6.1f}s +VRAM={meta['vram_peak_mib'] - meta['vram_base_mib']:5d}MiB "
            f"attempts={len(a)} done={[x.get('done_reason') for x in a]} prompt={[x.get('prompt_tokens') for x in a]} "
            f"gen={[x.get('gen_tokens') for x in a]}{' CONTENDED' if meta.get('other_gpu_jobs') else ''} -> {tail}")
