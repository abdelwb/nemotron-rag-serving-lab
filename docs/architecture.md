# Architecture & methodology

This doc is the detailed companion to the top-level [README](../README.md): exact steps, exact flags, what to expect at each stage, and why the pipeline is shaped the way it is.

## Stage 1 — Fine-tune (Hugging Face + PyTorch)

**Base model:** [`nvidia/Nemotron-Mini-4B-Instruct`](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) — an NVIDIA-published, distilled/pruned 4B model small enough to fine-tune on a free Colab T4 (16GB).

**Dataset:** [`nvidia/Daring-Anteater`](https://huggingface.co/datasets/nvidia/Daring-Anteater) — NVIDIA's own multi-turn instruction-tuning dataset. The notebook uses a small stratified subset (800 examples by default) rather than the full set, sized to actually finish in one free Colab session — see the sizing note below.

**Method:** supervised fine-tuning with LoRA via `peft` (rank 16, alpha 32, targeting the attention projection layers), 4-bit quantized base weights via `bitsandbytes` so the whole thing fits on 16GB, gradient checkpointing to keep activation memory down, `trl.SFTTrainer` driving the loop, `accelerate` under the hood.

Steps (see [`finetune/01_prepare_dataset.ipynb`](../finetune/01_prepare_dataset.ipynb) and [`finetune/02_lora_finetune.ipynb`](../finetune/02_lora_finetune.ipynb)):

1. Open both notebooks in Google Colab, select **Runtime → Change runtime type → T4 GPU**.
2. Run `huggingface_hub.login()` — needs a free Hugging Face account and a [write-scoped access token](https://huggingface.co/settings/tokens). Accept `Nemotron-Mini-4B-Instruct`'s model-card terms on the Hub first if it's gated.
3. `01_prepare_dataset.ipynb` downloads `nvidia/Daring-Anteater`, filters/formats it into the model's chat template, and saves a train/eval split.
4. `02_lora_finetune.ipynb` loads the base model 4-bit, attaches a LoRA adapter, trains for 1 epoch over the subset, evaluates loss on the held-out split, then **pushes just the adapter** (a few hundred MB) to your own `<your-hf-username>/nemotron-mini-4b-daring-anteater-lora` repo on the Hub.
5. **Sizing / expected wall-clock:** a free T4 running a 4B model in 4-bit with LoRA is genuinely compute-constrained, not just memory-constrained — at `per_device_train_batch_size=1` with gradient checkpointing, a single sequence's forward+backward is on the order of several seconds, so the total run time scales directly with `SUBSET_SIZE` (in notebook 1) × `num_train_epochs` (in notebook 2). The shipped defaults (800 examples, 1 epoch, 256-token sequences, batch size 4) land around 48 optimizer steps — a realistic target for one Colab session. Raise `SUBSET_SIZE` or sequence length only if you have a paid/longer-lived GPU session; the notebook's progress bar shows a live ETA after the first few logged steps so you can catch an unrealistic estimate early.

Only the adapter is pushed, not a merged copy of the base model — this keeps your Hub storage small and is exactly the artifact both vLLM and SGLang can load on top of the base model at serve time.

## Stage 2 — Serve and benchmark (vLLM vs. SGLang)

Both engines need a Linux host with an NVIDIA GPU (a cloud GPU instance, RunPod/Lambda, or a Colab instance kept alive via a terminal — a local install is not required and this repo's own dev machine does not have one).

**vLLM** ([`serving/vllm/`](../serving/vllm/)):
```bash
pip install vllm
bash serving/vllm/serve_vllm.sh   # launches an OpenAI-compatible server with --enable-lora
python serving/vllm/bench_vllm.py --base-url http://localhost:8000/v1 --out ../results/vllm_bench.csv
```

**SGLang** ([`serving/sglang/`](../serving/sglang/)):
```bash
pip install "sglang[all]"
bash serving/sglang/serve_sglang.sh
python serving/sglang/bench_sglang.py --base-url http://localhost:30000/v1 --out ../results/sglang_bench.csv
```

Both bench scripts sweep the same grid — batch size × (input length, output length) — against the same prompts, so the two CSVs are directly comparable. Run `serving/results/benchmark_report.md`'s companion script (documented in that file) once both CSVs exist to regenerate the comparison table in the top-level README.

`serving/sglang/structured_gen_demo.py` is a separate, smaller demo: constrained/JSON-mode generation exercising SGLang's compressed-FSM structured decoding — not part of the throughput sweep, just a correctness/latency demo of a feature vLLM doesn't do the same way.

## Stage 3 — RAG agent (LangChain)

`rag_agent/ingest.py` builds a local FAISS index over whatever documents you point it at (the `--docs-dir` flag — point it at this repo's own `docs/` folder to start, or your own notes/PDFs). It deliberately does **not** ship a pre-built corpus of NVIDIA/CUDA documentation in this repo — that content is NVIDIA's, and the point of `ingest.py` is that anyone can point it at their own docs at run time.

`rag_agent/router_baseline_sklearn.py` is a TF-IDF + logistic-regression classifier trained on a small labeled set of example queries (shipped inline in the script) that decides whether a query needs the retrieval+LLM path at all, versus being answerable directly or being out of scope. This is the "use a simpler model when a simpler model is enough" piece of the project.

`rag_agent/agent.py` wires it together: router → (retriever → ) LangChain agent with tool-calling, talking to whichever server is up over its OpenAI-compatible `/v1/chat/completions` endpoint — so the same agent code runs unmodified against vLLM or SGLang, just by changing `--base-url`.

## Stage 4 — Performance modeling (scikit-learn + TensorFlow)

Once `serving/results/vllm_bench.csv` and `sglang_bench.csv` have real rows:

```bash
python perf_modeling/train_sklearn_regressor.py   # GradientBoostingRegressor baseline
python perf_modeling/train_tf_model.py            # small Keras MLP
python perf_modeling/compare_models.py            # side-by-side predictions + error metrics
```

Both models are trained on the same features (`engine`, `batch_size`, `input_len`, `output_len`) predicting the same two targets (`throughput_tok_s`, `p99_latency_ms`). The comparison script reports MAE/RMSE for both so the README's results table reflects which model actually generalizes better on this project's own data, rather than assuming the deep-learning model wins by default.
