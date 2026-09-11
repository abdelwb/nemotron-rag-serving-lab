"""CPU-only tests for the shared benchmark sweep helpers (serving/bench_common.py).

No network calls -- these test the prompt generation and CSV writing only,
not the actual sweep against a live vLLM/SGLang server.
"""
import csv

from bench_common import BATCH_SIZES, SEQ_LENS, make_prompt, write_csv


def test_make_prompt_word_count():
    prompt = make_prompt(10)
    assert len(prompt.split()) == 10


def test_make_prompt_is_deterministic():
    assert make_prompt(20) == make_prompt(20)


def test_write_csv_roundtrip(tmp_path):
    rows = [{
        "engine": "vllm", "batch_size": 1, "input_len": 128, "output_len": 128,
        "wall_time_s": 1.0, "throughput_tok_s": 128.0, "p50_latency_ms": 10.0, "p99_latency_ms": 15.0,
    }]
    out_path = tmp_path / "bench.csv"
    write_csv(rows, str(out_path))

    with out_path.open() as f:
        read_rows = list(csv.DictReader(f))
    assert read_rows[0]["engine"] == "vllm"
    assert float(read_rows[0]["throughput_tok_s"]) == 128.0


def test_sweep_grid_is_nonempty():
    assert len(BATCH_SIZES) > 0
    assert len(SEQ_LENS) > 0
