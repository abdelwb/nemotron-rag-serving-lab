"""CPU-only, no-network tests for the sklearn query router (rag_agent/router_baseline_sklearn.py)."""
from router_baseline_sklearn import EXAMPLES, build_pipeline, classify


def _fitted_pipeline():
    pipeline = build_pipeline()
    texts = [t for t, _ in EXAMPLES]
    labels = [label for _, label in EXAMPLES]
    pipeline.fit(texts, labels)
    return pipeline


def test_classify_returns_a_known_label():
    pipeline = _fitted_pipeline()
    label = classify("How do I attach a LoRA adapter when serving with vLLM?", pipeline=pipeline)
    assert label in {"retrieve", "direct", "out_of_scope"}


def test_small_talk_routes_direct_not_retrieve():
    pipeline = _fitted_pipeline()
    assert classify("Hi, how are you?", pipeline=pipeline) == "direct"


def test_training_examples_are_labeled_consistently():
    # Guards against a future edit accidentally introducing a duplicate
    # query with two different labels.
    seen = {}
    for text, label in EXAMPLES:
        assert seen.setdefault(text, label) == label
