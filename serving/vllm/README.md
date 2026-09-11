# vLLM serving

```bash
pip install vllm
HF_LORA_REPO=<your-hf-username>/nemotron-mini-4b-daring-anteater-lora bash serve_vllm.sh
```

This starts an OpenAI-compatible server on `:8000` serving `nvidia/Nemotron-Mini-4B-Instruct` with the LoRA adapter attached under the name `nemotron-lora`.

In another shell, run the benchmark sweep:

```bash
pip install openai
python bench_vllm.py --base-url http://localhost:8000/v1 --model nemotron-lora --out ../results/vllm_bench.csv
```

This exercises `BATCH_SIZES x SEQ_LENS` from [`../bench_common.py`](../bench_common.py) and writes one row per configuration (throughput, p50/p99 latency) to the CSV. Run [`../sglang/bench_sglang.py`](../sglang/bench_sglang.py) against an SGLang server the same way to get a directly comparable `sglang_bench.csv`.

Requires a Linux host with an NVIDIA GPU — vLLM is not installed/runnable on this project's own dev machine (see [`../../docs/architecture.md`](../../docs/architecture.md) for where to run this: a Colab instance, RunPod, Lambda, or any CUDA box).
