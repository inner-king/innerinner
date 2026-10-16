#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fetch the Isaac Sim 5.1.0 wheels that BEHAVIOR-1K's setup.sh installs, resumably, into a persistent
# directory. Downloads over this link break off now and then (the fetch log has connection resets), so
# each wheel is fetched with `curl -C -` (resume), up to 8 attempts, and counts once it opens as a zip.
# They were then installed from that directory with `pip install ./<wheel> ...` in the simfoundry env.
# Usage: local/fetch_isaac_wheels.sh [dir] [list]
#        (default dir ~/isaac_wheels, list local/isaac_wheels.txt; local/isaac_wheels_extra.txt holds
#        the two ROS bridge wheels, fetched the same way)
set -uo pipefail
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
DIR="${1:-${HOME}/isaac_wheels}"
LIST="${2:-${HERE}/isaac_wheels.txt}"
mkdir -p "${DIR}" && cd "${DIR}"
fetch() {
  pkg="$1"; name="${pkg%-*}"; f="${pkg}-cp311-none-manylinux_2_35_x86_64.whl"
  for i in 1 2 3 4 5 6 7 8; do
    curl -fsSL --retry 5 -C - -o "$f" "https://pypi.nvidia.com/${name//_/-}/$f" && python3 -I -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).testzip()" "$f" && { echo "ok $f"; return 0; }
    echo "retry $i $f"; sleep 5
  done
  echo "FAILED $f"; return 1
}
export -f fetch
xargs -P 4 -I{} bash -c 'fetch {}' < "${LIST}"
