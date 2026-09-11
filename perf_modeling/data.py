"""Shared data loading for the throughput/latency prediction models.

Loads serving/results/vllm_bench.csv and sglang_bench.csv, one-hot encodes
the categorical `engine` column, and returns arrays ready for either
scikit-learn or TensorFlow. Kept in one place so both models train on
exactly the same feature encoding -- otherwise compare_models.py's
side-by-side comparison wouldn't be a fair one.
"""
import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "serving" / "results"
FEATURE_COLUMNS = ["batch_size", "input_len", "output_len"]
TARGET_COLUMNS = ["throughput_tok_s", "p99_latency_ms"]
MIN_ROWS = 10


def load_bench_data() -> pd.DataFrame:
    frames = []
    for name in ("vllm_bench.csv", "sglang_bench.csv"):
        path = RESULTS_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            if len(df):
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["engine", *FEATURE_COLUMNS, *TARGET_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def require_enough_data(df: pd.DataFrame) -> None:
    if len(df) < MIN_ROWS:
        sys.exit(
            f"Only {len(df)} benchmark rows found (need >= {MIN_ROWS}).\n"
            "Run serving/vllm/bench_vllm.py and serving/sglang/bench_sglang.py "
            "against live servers first -- see docs/architecture.md."
        )


def encode_features(df: pd.DataFrame):
    """One-hot encode `engine`, keep the numeric columns as-is.

    Returns (X, y, feature_names) as numpy arrays plus the encoded column
    order -- compare_models.py needs that order to encode a single new row
    for a `--predict`-style query the same way.
    """
    encoded = pd.get_dummies(df[["engine", *FEATURE_COLUMNS]], columns=["engine"])
    x = encoded.to_numpy(dtype="float32")
    y = df[TARGET_COLUMNS].to_numpy(dtype="float32")
    return x, y, list(encoded.columns)
