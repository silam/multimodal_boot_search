"""Eval harness: runs the golden set and reports retrieval hit rate + answer checks.

Run:  uv run python -m evals.run_evals            (uses DOCQA_LLM_PROVIDER from env/.env)
Exit code is non-zero if scores fall below thresholds -> usable as a CI gate.
"""

import argparse
import json
import sys
from pathlib import Path

from docqa.config import get_settings
from docqa.services.container import build_qa_service

DATASET = Path(__file__).parent / "datasets" / "qa_golden.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--min-retrieval", type=float, default=0.8)
    parser.add_argument("--min-answer", type=float, default=0.6)
    args = parser.parse_args()

    settings = get_settings()
    qa = build_qa_service(settings)
    cases = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]

    retrieval_hits = answer_hits = 0
    for case in cases:
        answer, hits = qa.ask(case["question"])
        sources = [h.chunk.source for h in hits]
        r_ok = bool(sources) and sources[0] == case["expected_source"]
        a_ok = all(s.lower() in answer.answer.lower() for s in case["must_contain"])
        retrieval_hits += r_ok
        answer_hits += a_ok
        mark = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
        print(f"[retrieval {mark(r_ok)}] [answer {mark(a_ok)}] {case['question']}")
        print(f"    -> {answer.answer}")

    n = len(cases)
    r_score, a_score = retrieval_hits / n, answer_hits / n
    print(f"\nprovider={settings.llm_provider} prompt={settings.answer_prompt}")
    print(f"retrieval@1: {r_score:.0%}   answer: {a_score:.0%}   (n={n})")
    if settings.llm_provider == "fake":
        # The fake LLM only exercises plumbing; answer quality is gated with a real model.
        print("note: fake provider -> gating on retrieval only")
        return 0 if r_score >= args.min_retrieval else 1
    return 0 if r_score >= args.min_retrieval and a_score >= args.min_answer else 1


if __name__ == "__main__":
    sys.exit(main())
