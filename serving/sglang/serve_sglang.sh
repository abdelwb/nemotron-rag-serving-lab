#!/usr/bin/env bash
# Launch SGLang's OpenAI-compatible server serving nvidia/Nemotron-Mini-4B-Instruct
# with the fine-tuned LoRA adapter from finetune/02_lora_finetune.ipynb attached.
#
# Needs a Linux host with an NVIDIA GPU (SGLang does not run on this project's
# own Windows dev machine -- see docs/architecture.md for where to run this).
#
# Usage:
#   pip install "sglang[all]"
#   HF_LORA_REPO=<your-hf-username>/nemotron-mini-4b-daring-anteater-lora bash serve_sglang.sh

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-nvidia/Nemotron-Mini-4B-Instruct}"
LORA_REPO="${HF_LORA_REPO:?Set HF_LORA_REPO to your pushed adapter repo id, e.g. you/nemotron-mini-4b-daring-anteater-lora}"
PORT="${PORT:-30000}"

python -m sglang.launch_server \
  --model-path "$BASE_MODEL" \
  --lora-paths "nemotron-lora=${LORA_REPO}" \
  --port "$PORT" \
  --dtype bfloat16
