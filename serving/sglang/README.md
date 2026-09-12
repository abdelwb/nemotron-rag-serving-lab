# SGLang serving

```bash
pip install "sglang[all]"
HF_LORA_REPO=abdelwb/nemotron-mini-4b-daring-anteater-lora bash serve_sglang.sh
```

This starts an OpenAI-compatible server on `:30000` serving `nvidia/Nemotron-Mini-4B-Instruct` with the LoRA adapter attached under the name `nemotron-lora`.

**Benchmark** (same sweep as vLLM, for a direct comparison):

```bash
pip install openai
python bench_sglang.py --base-url http://localhost:30000/v1 --model nemotron-lora --out ../results/sglang_bench.csv
```

**Structured generation demo** (SGLang-specific - not part of the throughput sweep):

```bash
python structured_gen_demo.py --base-url http://localhost:30000
```

Uses SGLang's native frontend (`sgl.function` / `sgl.gen(..., regex=...)`) to constrain generation to valid JSON matching an exact schema, with a shared system prompt across calls so RadixAttention's prefix cache gets reused instead of recomputed. See the docstring in [`structured_gen_demo.py`](structured_gen_demo.py) for a note on API stability.

Requires a Linux host with an NVIDIA GPU - SGLang is not installed/runnable on this project's own dev machine (see [`../../docs/architecture.md`](../../docs/architecture.md) for where to run this).
