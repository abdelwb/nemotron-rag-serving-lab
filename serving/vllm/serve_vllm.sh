#!/usr/bin/env bash
# Launch an OpenAI-compatible vLLM server serving nvidia/Nemotron-Mini-4B-Instruct
# with the fine-tuned LoRA adapter from finetune/02_lora_finetune.ipynb attached.
#
# Needs a Linux host with an NVIDIA GPU (vLLM does not run on this project's
# own Windows dev machine -- see docs/architecture.md for where to run this).
#
# Usage:
#   pip install vllm
#   HF_LORA_REPO=abdelwb/nemotron-mini-4b-daring-anteater-lora bash serve_vllm.sh

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-nvidia/Nemotron-Mini-4B-Instruct}"
LORA_REPO="${HF_LORA_REPO:?Set HF_LORA_REPO to your pushed adapter repo id, e.g. you/nemotron-mini-4b-daring-anteater-lora}"
PORT="${PORT:-8000}"

vllm serve "$BASE_MODEL" \
  --port "$PORT" \
  --enable-lora \
  --lora-modules "nemotron-lora=${LORA_REPO}" \
  --max-lora-rank 16 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90
