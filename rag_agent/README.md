# RAG agent

A LangChain retrieval + tool-calling agent that talks to whichever server from [`../serving/`](../serving/) is running, over its OpenAI-compatible endpoint. Same agent code, either engine - just change `--base-url`.

```bash
pip install -r requirements.txt

# 1. Build the index (defaults to indexing this repo's own docs)
python ingest.py --docs-dir ../docs --index-dir ./faiss_index

# 2. Check retrieval quality on a small labeled query set
python eval_retrieval.py --index-dir ./faiss_index

# 3. Ask the agent something, against a running vLLM server...
python agent.py --base-url http://localhost:8000/v1 --model nemotron-lora \
    --index-dir ./faiss_index --query "What batch sizes were swept in the vLLM benchmark?"

# ...or the identical command against SGLang instead:
python agent.py --base-url http://localhost:30000/v1 --model nemotron-lora \
    --index-dir ./faiss_index --query "What batch sizes were swept in the vLLM benchmark?"
```

## Files

- [`ingest.py`](ingest.py) - chunks Markdown docs and builds a local FAISS index using a small `sentence-transformers` embedding model (no API key, no running LLM needed for this step).
- [`router_baseline_sklearn.py`](router_baseline_sklearn.py) - TF-IDF + logistic-regression classifier deciding `retrieve` / `direct` / `out_of_scope` before spending an LLM call. Run standalone to retrain and see a classification report.
- [`eval_retrieval.py`](eval_retrieval.py) - grades the FAISS index's ranking quality with scikit-learn's `ndcg_score` against a small hand-labeled query set.
- [`agent.py`](agent.py) - the actual agent: router gate → (retriever tool + a calculator tool) → tool-calling LLM via `langchain_openai.ChatOpenAI` pointed at vLLM or SGLang.

Retrieval defaults to indexing this repo's own `docs/` folder as a placeholder corpus - point `--docs-dir` at your own documents (e.g. your own notes on CUDA/GPU performance topics) for a more realistic demo. No third-party documentation is bundled in this repo.
