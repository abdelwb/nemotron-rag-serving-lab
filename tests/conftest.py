"""Put the standalone script modules (rag_agent/, serving/, perf_modeling/)
on sys.path so tests can `import` them directly -- none of these are
packaged with __init__.py on purpose, since each is meant to be run as a
plain script from its own directory.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for sub in ("rag_agent", "serving", "perf_modeling"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
