#!/usr/bin/env python
"""Retrieval-quality evaluation for the FAISS index built by ingest.py.

Uses a small hand-labeled set of (query, relevant-chunk substrings) pairs
and scikit-learn's `ndcg_score` to grade ranking quality -- not just whether
*a* relevant chunk was retrieved, but whether it was ranked near the top.

    python eval_retrieval.py --index-dir ./faiss_index
"""
import argparse

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics import ndcg_score

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Each query maps to substrings that should appear in a *relevant* retrieved
# chunk. These were written against this repo's own docs/ and README.md --
# adjust once you've pointed ingest.py at your own documents.
LABELED_QUERIES = [
    ("What license is the Nemotron model under?", ["license", "NVIDIA"]),
    ("How is the benchmark sweep structured?", ["batch", "sweep", "throughput"]),
    ("What does the sklearn router decide?", ["router", "retrieve", "sklearn"]),
]

K = 4


def score_query(vectorstore, query: str, relevant_substrings: list) -> float:
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=K)
    if not docs_and_scores:
        return 0.0

    # 1.0 if a labeled substring appears in the chunk, 0.0 otherwise.
    relevance = [
        1.0 if any(s.lower() in doc.page_content.lower() for s in relevant_substrings) else 0.0
        for doc, _ in docs_and_scores
    ]
    if sum(relevance) == 0:
        return 0.0  # nothing relevant retrieved at all -- nDCG is 0, not undefined

    # FAISS reports L2 distance (lower = better); negate for an nDCG "gain" score.
    predicted_gain = [-score for _, score in docs_and_scores]
    return float(ndcg_score([relevance], [predicted_gain]))


def main(args):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(args.index_dir, embeddings, allow_dangerous_deserialization=True)

    scores = []
    for query, relevant in LABELED_QUERIES:
        score = score_query(vectorstore, query, relevant)
        scores.append(score)
        print(f"nDCG@{K}={score:.3f}  <-  {query}")

    print(f"\nMean nDCG@{K}: {np.mean(scores):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="./faiss_index")
    main(parser.parse_args())
