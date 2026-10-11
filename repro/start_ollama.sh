#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Local Ollama server backing the SimFoundry Gemini shim (text/vision requests).
# Models unload after OLLAMA_KEEP_ALIVE so the 12 GB GPU is free for pipeline stages.
#
#   OLLAMA_BIN     ollama executable (default: `ollama` on PATH)
#   OLLAMA_MODELS  model store holding qwen3-vl:8b-instruct (default: Ollama's own)
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_KEEP_ALIVE=60s
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_CONTEXT_LENGTH=16384
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
exec "${OLLAMA_BIN:-ollama}" serve
