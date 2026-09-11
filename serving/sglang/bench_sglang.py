#!/usr/bin/env python
"""Throughput/latency sweep against the SGLang server started by serve_sglang.sh.

    python bench_sglang.py --base-url http://localhost:30000/v1 --out ../results/sglang_bench.csv

Runs the identical sweep as bench_vllm.py (see serving/bench_common.py) so
the two output CSVs are directly comparable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench_common import main_sync  # noqa: E402

if __name__ == "__main__":
    main_sync(
        default_base_url="http://localhost:30000/v1",
        default_engine="sglang",
        default_out="../results/sglang_bench.csv",
    )
