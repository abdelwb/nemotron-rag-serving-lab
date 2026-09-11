"""CPU-only tests for perf_modeling/data.py using synthetic benchmark rows --
no dependency on a real vLLM/SGLang run.
"""
import data as perf_data
import pandas as pd
import pytest


@pytest.fixture
def synthetic_bench_dir(tmp_path, monkeypatch):
    vllm_df = pd.DataFrame([
        {"engine": "vllm", "batch_size": 1, "input_len": 128, "output_len": 128,
         "wall_time_s": 1.0, "throughput_tok_s": 100.0, "p50_latency_ms": 10.0, "p99_latency_ms": 12.0},
        {"engine": "vllm", "batch_size": 4, "input_len": 128, "output_len": 128,
         "wall_time_s": 1.5, "throughput_tok_s": 300.0, "p50_latency_ms": 15.0, "p99_latency_ms": 20.0},
    ])
    sglang_df = pd.DataFrame([
        {"engine": "sglang", "batch_size": 1, "input_len": 128, "output_len": 128,
         "wall_time_s": 0.9, "throughput_tok_s": 110.0, "p50_latency_ms": 9.0, "p99_latency_ms": 11.0},
    ])
    vllm_df.to_csv(tmp_path / "vllm_bench.csv", index=False)
    sglang_df.to_csv(tmp_path / "sglang_bench.csv", index=False)

    monkeypatch.setattr(perf_data, "RESULTS_DIR", tmp_path)
    return tmp_path


def test_load_bench_data_concatenates_both_engines(synthetic_bench_dir):
    df = perf_data.load_bench_data()
    assert set(df["engine"]) == {"vllm", "sglang"}
    assert len(df) == 3


def test_encode_features_shapes(synthetic_bench_dir):
    df = perf_data.load_bench_data()
    x, y, feature_names = perf_data.encode_features(df)
    assert x.shape[0] == 3
    assert y.shape == (3, 2)
    assert "engine_vllm" in feature_names
    assert "engine_sglang" in feature_names


def test_require_enough_data_exits_when_too_few_rows():
    df = pd.DataFrame({"engine": ["vllm"]})
    with pytest.raises(SystemExit):
        perf_data.require_enough_data(df)
