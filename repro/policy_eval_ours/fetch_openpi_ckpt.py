# SPDX-License-Identifier: Apache-2.0
"""Download an openpi checkpoint the way run_local_openpi.sh expects it (repro/README.md, chapter 30).

Fetches gs://openpi-assets/checkpoints/<name> and the norm-stats folder of the matching <name>_polaris config into
$OPENPI_DATA_HOME, laid out as openpi's maybe_download lays them out, so the policy server finds both in the cache.
openpi's own download (gcsfs) stalled twice on this link with an open connection and no bytes moving, 11 GB into the
12 GB pi0.5 checkpoint; here gcsfs only lists the bucket and curl fetches each file, resuming where it stopped and
dropping a transfer that falls under 100 kB/s for 30 s. Anonymous, as setup_colab.sh does.

    OPENPI_DATA_HOME=~/.cache/openpi <openpi>/.venv/bin/python repro/policy_eval_ours/fetch_openpi_ckpt.py pi05_droid_jointpos
"""
import os
import pathlib
import subprocess
import sys

import gcsfs

ATTEMPTS = 40


def fetch(fs, remote_dir, final):
    """Every file under remote_dir into final (via final.partial, renamed once all sizes match)."""
    if final.exists():
        print(f"have {final}", flush=True)
        return
    part = final.with_name(final.name + ".partial")
    files = {p: i["size"] for p, i in fs.find(remote_dir, detail=True).items() if i["type"] == "file"}
    print(f"{remote_dir}: {len(files)} files, {sum(files.values()) / 1e9:.2f} GB", flush=True)
    for path, size in files.items():
        dst = part / path[len(remote_dir) + 1:]
        dst.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(ATTEMPTS):
            if dst.exists() and dst.stat().st_size == size:
                break
            if size == 0:
                dst.touch()
                break
            subprocess.run(["curl", "-sS", "-L", "-C", "-", "--speed-limit", "100000", "--speed-time", "30",
                            "-o", str(dst), f"https://storage.googleapis.com/{path}"])
        if not (dst.exists() and dst.stat().st_size == size):
            sys.exit(f"{path}: {dst.stat().st_size if dst.exists() else 0} of {size} bytes after {ATTEMPTS} attempts")
    part.rename(final)
    print(f"complete {final}", flush=True)


def main():
    name = sys.argv[1]
    base = pathlib.Path(os.environ["OPENPI_DATA_HOME"]) / "openpi-assets" / "checkpoints"
    fs = gcsfs.GCSFileSystem(token="anon")
    fetch(fs, f"openpi-assets/checkpoints/{name}", base / name)
    fetch(fs, f"openpi-assets/checkpoints/polaris/{name}_polaris/assets/droid",
          base / "polaris" / f"{name}_polaris" / "assets" / "droid")


if __name__ == "__main__":
    main()
