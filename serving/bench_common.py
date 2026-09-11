"""Shared throughput/latency sweep logic for the vLLM and SGLang benchmarks.

Both engines expose an OpenAI-compatible /v1/chat/completions endpoint, so the
exact same request/measurement code drives both `bench_vllm.py` and
`bench_sglang.py` -- only the base URL, default port, and the "engine" label
written into the output CSV differ. Keeping this shared makes the two CSVs a
fair, apples-to-apples comparison instead of two independently-drifting
scripts.
"""
import argparse
import asyncio
import csv
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI

# (batch_size == concurrent requests) x (input_len, output_len) in approx tokens
BATCH_SIZES = [1, 4, 8, 16, 32]
SEQ_LENS = [(128, 128), (512, 128), (128, 512)]

_WORDS = ("the quick brown fox jumps over the lazy dog near the river bank " * 100).split()


def make_prompt(n_tokens: int) -> str:
    """Rough word-approximates-token synthetic prompt.

    Good enough for a relative vLLM-vs-SGLang comparison; not a
    tokenizer-exact token count for either engine's tokenizer.
    """
    return " ".join(_WORDS[:n_tokens])


async def _run_one_request(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int):
    start = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    elapsed = time.perf_counter() - start
    completion_tokens = resp.usage.completion_tokens if resp.usage else max_tokens
    return elapsed, completion_tokens


async def _run_batch(client, model, batch_size, input_len, output_len):
    prompt = make_prompt(input_len)
    tasks = [_run_one_request(client, model, prompt, output_len) for _ in range(batch_size)]

    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - start

    latencies_ms = sorted(r[0] * 1000 for r in results)
    total_tokens = sum(r[1] for r in results)
    throughput = total_tokens / wall_time if wall_time > 0 else 0.0
    p99_idx = max(0, int(len(latencies_ms) * 0.99) - 1)

    return {
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "wall_time_s": round(wall_time, 3),
        "throughput_tok_s": round(throughput, 2),
        "p50_latency_ms": round(statistics.median(latencies_ms), 1),
        "p99_latency_ms": round(latencies_ms[p99_idx], 1),
    }


async def run_sweep(base_url: str, model: str, engine_label: str):
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    rows = []
    for batch_size in BATCH_SIZES:
        for input_len, output_len in SEQ_LENS:
            print(f"[{engine_label}] batch={batch_size} in={input_len} out={output_len} ...")
            row = await _run_batch(client, model, batch_size, input_len, output_len)
            row["engine"] = engine_label
            rows.append(row)
            print(
                f"  -> {row['throughput_tok_s']} tok/s | "
                f"p50={row['p50_latency_ms']}ms | p99={row['p99_latency_ms']}ms"
            )
    return rows


def write_csv(rows, out: str):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "engine", "batch_size", "input_len", "output_len",
        "wall_time_s", "throughput_tok_s", "p50_latency_ms", "p99_latency_ms",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


def build_arg_parser(default_base_url: str, default_engine: str, default_out: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=default_base_url)
    parser.add_argument("--model", default="nemotron-lora", help="Model/LoRA name as registered with the server")
    parser.add_argument("--engine", default=default_engine, help="Label written into the 'engine' column")
    parser.add_argument("--out", default=default_out)
    return parser


def main_sync(default_base_url: str, default_engine: str, default_out: str):
    args = build_arg_parser(default_base_url, default_engine, default_out).parse_args()
    rows = asyncio.run(run_sweep(args.base_url, args.model, args.engine))
    write_csv(rows, args.out)
