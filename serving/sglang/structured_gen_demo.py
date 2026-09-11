"""Structured/constrained generation demo using SGLang's native frontend.

This demonstrates the piece vLLM doesn't do the same way: SGLang's
compressed finite-state-machine constrained decoding, exercised here as a
JSON-shaped field-extraction task, plus a shared system prompt across every
call so RadixAttention's prefix cache is reused instead of recomputed each
time.

Not part of the throughput/latency sweep (see bench_sglang.py) -- this is a
correctness + "does structured output actually stay valid JSON" demo of a
feature that's central to SGLang's pitch.

NOTE: SGLang's frontend API moves fast. If `import sglang as sgl` /
`sgl.function` / `sgl.gen(..., regex=...)` has been renamed or reorganized
since this was written, check https://docs.sglang.ai for the current
constrained-decoding entry point -- the shape of the program (shared prefix
+ regex/JSON-constrained field extraction) is the durable part of this demo.
"""
import argparse
import json

import sglang as sgl

SHARED_SYSTEM_PROMPT = (
    "You are a terse extraction assistant. Given a short product review, "
    "extract structured fields. Respond with JSON only."
)

# Constrains decoding to exactly this JSON shape -- the model cannot emit
# malformed JSON or an out-of-range rating even if it "wants" to.
REVIEW_JSON_REGEX = (
    r'\{\s*"sentiment":\s*"(positive|negative|neutral)",\s*'
    r'"rating_1_to_5":\s*[1-5],\s*'
    r'"one_word_summary":\s*"[a-zA-Z]+"\s*\}'
)

SAMPLE_REVIEWS = [
    "This GPU runs cooler than I expected and the fan noise is barely noticeable. Would buy again.",
    "Arrived with a bent bracket and the drivers crashed twice on install. Not happy.",
    "It's fine. Does what the box says, nothing more, nothing less.",
]


@sgl.function
def extract_review(s, system_prompt, review):
    s += sgl.system(system_prompt)
    s += sgl.user(review)
    s += sgl.assistant(sgl.gen("extraction", regex=REVIEW_JSON_REGEX, max_tokens=64))


def main(base_url: str):
    sgl.set_default_backend(sgl.RuntimeEndpoint(base_url))

    for review in SAMPLE_REVIEWS:
        state = extract_review.run(system_prompt=SHARED_SYSTEM_PROMPT, review=review)
        raw = state["extraction"]
        parsed = json.loads(raw)  # will raise if the constraint somehow failed
        print(f"review: {review[:60]}...")
        print(f"  -> {parsed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:30000")
    args = parser.parse_args()
    main(args.base_url)
