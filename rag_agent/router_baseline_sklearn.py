#!/usr/bin/env python
"""A cheap scikit-learn gate in front of the LLM/retrieval path.

Not every query needs retrieval, and not every query even needs the LLM --
a TF-IDF + logistic-regression classifier that decides "retrieve", "direct",
or "out_of_scope" costs microseconds and no GPU, versus a full agent turn.
This is a small, honest example of using a simpler model where a simpler
model is genuinely enough.

    python router_baseline_sklearn.py            # trains, prints a classification report, saves router.joblib
    python router_baseline_sklearn.py --query "..."   # loads a saved router and classifies one query
"""
import argparse
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Small hand-labeled example set. In a real deployment this would grow from
# logged, human-reviewed traffic -- shipped inline here so the script is
# self-contained and reproducible without an external label file.
EXAMPLES = [
    ("What's the difference between PagedAttention and RadixAttention?", "retrieve"),
    ("How do I attach a LoRA adapter when serving with vLLM?", "retrieve"),
    ("What dataset was this model fine-tuned on?", "retrieve"),
    ("Summarize the benchmark methodology used in this repo.", "retrieve"),
    ("What batch sizes were swept in the vLLM benchmark?", "retrieve"),
    ("Explain how the FAISS index is built in ingest.py.", "retrieve"),
    ("What license is the Nemotron model under?", "retrieve"),
    ("Hi, how are you?", "direct"),
    ("What's 2 + 2?", "direct"),
    ("Write a haiku about GPUs.", "direct"),
    ("Translate 'good morning' to French.", "direct"),
    ("Tell me a joke.", "direct"),
    ("What's the capital of France?", "direct"),
    ("What's the weather in Tokyo right now?", "out_of_scope"),
    ("Can you place an order for a new graphics card?", "out_of_scope"),
    ("What's my account balance?", "out_of_scope"),
    ("Book me a flight to San Francisco.", "out_of_scope"),
    ("What's the current stock price of NVIDIA?", "out_of_scope"),
]

MODEL_PATH = Path(__file__).parent / "router.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train_and_evaluate():
    texts = [t for t, _ in EXAMPLES]
    labels = [l for _, l in EXAMPLES]

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    preds = pipeline.predict(x_test)
    print(classification_report(y_test, preds, zero_division=0))

    # Refit on the full set before saving -- the split above is just to sanity-check it.
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved router to {MODEL_PATH}")
    return pipeline


def classify(query: str, pipeline: Pipeline | None = None) -> str:
    if pipeline is None:
        if not MODEL_PATH.exists():
            pipeline = train_and_evaluate()
        else:
            pipeline = joblib.load(MODEL_PATH)
    return pipeline.predict([query])[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=None, help="Classify a single query instead of (re)training")
    args = parser.parse_args()

    if args.query:
        label = classify(args.query)
        print(f"{args.query!r} -> {label}")
    else:
        train_and_evaluate()
