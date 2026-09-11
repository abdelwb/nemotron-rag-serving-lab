"""Throughput/latency sweep against the vLLM server started by serve_vllm.sh.

    python bench_vllm.py --base-url http://localhost:8000/v1 --out ../results/vllm_bench.csv

See serving/bench_common.py for the shared sweep logic (SGLang's
bench_sglang.py runs the identical sweep so the two CSVs are comparable).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench_common import main_sync

if __name__ == "__main__":
    main_sync(
        default_base_url="http://localhost:8000/v1",
        default_engine="vllm",
        default_out="../results/vllm_bench.csv",
    )
