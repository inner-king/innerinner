# SPDX-License-Identifier: Apache-2.0
# This machine (host "super": i9-12900K, RTX 3060 12 GB, Ubuntu 24.04). The repro/ wrappers source
# this file when it exists; it only fills in values that are not already set. See local/README.md.
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniforge3}"
# Ollama is a standalone install next to the checkouts; its model store was moved to /data.
export OLLAMA_BIN="${OLLAMA_BIN:-/home/inner/real2sim/tools/ollama/bin/ollama}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-/data/models/ollama}"
export SIMFOUNDRY_LOG_DIR="${SIMFOUNDRY_LOG_DIR:-${HOME}/simfoundry_logs}"
# rclone remote "innerking" has the Inner-King Drive folder as its root (~/.config/rclone/rclone.conf).
export INNERINNER_DRIVE="${INNERINNER_DRIVE:-innerking:innerinner}"
# The first v1 runs used SAM 3.1 detector weights (repro/README.md, chapter 2); to reproduce those:
#   export SIMFOUNDRY_SAM3_CHECKPOINT=/data/models/real2sim-tools-checkpoints/sam3.1_multiplex_fp16.pt
# `git clone` over this link drops TLS mid-transfer now and then (rp.git_import clones at run time);
# local/bin/git retries clones and passes everything else to /usr/bin/git.
_local_bin="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bin"
case ":${PATH}:" in
  *":${_local_bin}:"*) ;;
  *) export PATH="${_local_bin}:${PATH}" ;;
esac
unset _local_bin
