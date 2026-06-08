"""Offline evaluation for vigilant-rag — retrieval metrics + LLM-as-judge.

Usage:
    python -m src.evaluation.evaluator
    python -m src.evaluation.evaluator --judge --output results.json
    python -m src.evaluation.evaluator --difficulty easy --top-k 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from src.evaluation.dataset import EvalEntry, load_dataset

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "of", "and", "or", "but", "with", "by", "from", "as", "be",
    "have", "has", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "may", "might", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "its",
    "what", "which", "who", "whom", "when", "where", "how", "why", "if",
})


# ── Metric dataclasses ────────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hits: int
    total_expected: int


@dataclass
class EvalResult:
    query: str
    mode: str
    difficulty: str
    answer: str
    returned_sources: list[str]
    retrieval: RetrievalMetrics
    keyword_coverage: float
    hallucination_rate: float
    llm_score: float | None
    latency_ms: int
    error: str | None = None


# ── Metric functions ──────────────────────────────────────────────────────────

def _source_matches(returned: str, expected: str) -> bool:
    """True when the returned path ends with the expected path suffix."""
    r = returned.replace("\\", "/")
    e = expected.replace("\\", "/")
    return r.endswith(e) or r == e


def compute_retrieval_metrics(
    returned: list[str],
    expected: list[str],
    k: int,
) -> RetrievalMetrics:
    top_k = returned[:k]
    hits = len({r for r in top_k if any(_source_matches(r, e) for e in expected)})
    precision = hits / k if k > 0 else 0.0
    recall = min(hits / len(expected), 1.0) if expected else 1.0
    mrr = 0.0
    for rank, r in enumerate(top_k, 1):
        if any(_source_matches(r, e) for e in expected):
            mrr = 1.0 / rank
            break
    return RetrievalMetrics(
        precision_at_k=precision,
        recall_at_k=recall,
        mrr=mrr,
        hits=hits,
        total_expected=len(expected),
    )


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    lower = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in lower) / len(keywords)


def hallucination_rate(answer: str, context_chunks: list[str]) -> float:
    """Fraction of answer content-words not found in any retrieved chunk."""
    context_text = " ".join(context_chunks)
    answer_words = {w for w in re.findall(r"\b\w+\b", answer.lower()) if w not in _STOPWORDS and len(w) > 2}
    context_words = set(re.findall(r"\b\w+\b", context_text.lower()))
    if not answer_words:
        return 0.0
    novel = answer_words - context_words
    return len(novel) / len(answer_words)


async def llm_judge(
    query: str,
    answer: str,
    excerpts: list[str],
    ollama_url: str,
    model: str,
) -> float:
    context = "\n---\n".join(ex[:400] for ex in excerpts[:3])
    prompt = (
        "You are evaluating a RAG system answer. Score 1–5.\n\n"
        f"Query: {query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Answer: {answer}\n\n"
        "Rubric:\n"
        "1=completely wrong  2=major gaps  3=partial  4=good  5=excellent\n\n"
        "Reply with ONLY the integer (1-5)."
    )
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        text = resp.json()["response"].strip()
    for ch in text:
        if ch.isdigit() and ch != "0":
            return float(ch)
    return 0.0


# ── Per-entry evaluation ──────────────────────────────────────────────────────

async def evaluate_entry(
    entry: EvalEntry,
    *,
    api_url: str,
    ollama_url: str,
    llm_model: str,
    k: int,
    run_judge: bool,
) -> EvalResult:
    empty_metrics = RetrievalMetrics(0.0, 0.0, 0.0, 0, len(entry.expected_sources))
    t0 = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{api_url}/api/v1/query",
                json={"query": entry.query, "mode": entry.mode, "top_k": k},
            )
            resp.raise_for_status()
    except Exception as exc:
        latency = int((time.monotonic() - t0) * 1000)
        return EvalResult(
            query=entry.query, mode=entry.mode, difficulty=entry.difficulty,
            answer="", returned_sources=[], retrieval=empty_metrics,
            keyword_coverage=0.0, hallucination_rate=1.0,
            llm_score=None, latency_ms=latency, error=str(exc),
        )

    latency = int((time.monotonic() - t0) * 1000)
    data = resp.json()
    answer = data.get("answer", "")
    sources = [s["file_path"] for s in data.get("sources", [])]
    excerpts = [s.get("excerpt", "") for s in data.get("sources", [])]

    ret = compute_retrieval_metrics(sources, entry.expected_sources, k)
    kw = keyword_coverage(answer, entry.expected_answer_contains)
    hall = hallucination_rate(answer, excerpts)

    score: float | None = None
    if run_judge and answer:
        try:
            score = await llm_judge(entry.query, answer, excerpts, ollama_url, llm_model)
        except Exception:
            pass

    return EvalResult(
        query=entry.query, mode=entry.mode, difficulty=entry.difficulty,
        answer=answer, returned_sources=sources,
        retrieval=ret, keyword_coverage=kw, hallucination_rate=hall,
        llm_score=score, latency_ms=latency,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(results: list[EvalResult], k: int) -> None:
    ok = [r for r in results if not r.error]
    failed = [r for r in results if r.error]

    if not ok:
        print("All queries failed — no metrics to report.")
        for r in failed:
            print(f"  {r.query[:60]}: {r.error}")
        return

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    prec = avg([r.retrieval.precision_at_k for r in ok])
    rec  = avg([r.retrieval.recall_at_k for r in ok])
    mrr  = avg([r.retrieval.mrr for r in ok])
    kw   = avg([r.keyword_coverage for r in ok])
    hall = avg([r.hallucination_rate for r in ok])
    lat  = avg([r.latency_ms for r in ok])
    scores = [r.llm_score for r in ok if r.llm_score is not None]

    W = 54
    print("\n" + "=" * W)
    print(f"  vigilant-rag Eval  —  {len(results)} queries  top-{k}")
    print("=" * W)
    print(f"  Precision@{k:<2}         {prec:.3f}")
    print(f"  Recall@{k:<2}            {rec:.3f}")
    print(f"  MRR                  {mrr:.3f}")
    print(f"  Keyword coverage     {kw:.3f}")
    hall_flag = "  ⚠ high" if hall > 0.30 else ""
    print(f"  Hallucination rate   {hall:.3f}{hall_flag}")
    if scores:
        print(f"  LLM-judge avg        {avg(scores):.2f} / 5.0  (n={len(scores)})")
    print(f"  Avg latency          {lat:.0f} ms")
    if failed:
        print(f"  Errors               {len(failed)} / {len(results)}")
    print("=" * W)

    # Per-mode breakdown
    modes = sorted({r.mode for r in ok})
    if len(modes) > 1:
        print(f"\n  Per-mode (P@{k} / MRR):")
        for mode in modes:
            sub = [r for r in ok if r.mode == mode]
            mp = avg([r.retrieval.precision_at_k for r in sub])
            mm = avg([r.retrieval.mrr for r in sub])
            print(f"    {mode:<14}  P@{k}={mp:.2f}  MRR={mm:.2f}  n={len(sub)}")

    # Per-difficulty breakdown
    diffs = sorted({r.difficulty for r in ok})
    if len(diffs) > 1:
        print(f"\n  Per-difficulty (P@{k} / MRR):")
        for diff in diffs:
            sub = [r for r in ok if r.difficulty == diff]
            mp = avg([r.retrieval.precision_at_k for r in sub])
            mm = avg([r.retrieval.mrr for r in sub])
            print(f"    {diff:<14}  P@{k}={mp:.2f}  MRR={mm:.2f}  n={len(sub)}")

    # Per-result table
    print(f"\n  {'Query':<38} {'Mode':<10} P@k   MRR   KW  {'ms':>6}")
    print(f"  {'-'*38} {'-'*10} {'----':>4} {'----':>5} {'----':>4} {'------':>6}")
    for r in results:
        q = (r.query[:35] + "...") if len(r.query) > 38 else r.query
        if r.error:
            print(f"  {q:<38} {r.mode:<10} ERROR  ({r.error[:30]})")
        else:
            p  = f"{r.retrieval.precision_at_k:.2f}"
            m  = f"{r.retrieval.mrr:.2f}"
            kw = f"{r.keyword_coverage:.2f}"
            print(f"  {q:<38} {r.mode:<10} {p:>4}  {m:>4}  {kw:>4}  {r.latency_ms:>6}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> list[EvalResult]:
    path = Path(args.dataset)
    if not path.exists():
        print(f"Dataset not found: {path}", file=sys.stderr)
        sys.exit(1)

    entries = load_dataset(path)
    print(f"Loaded {len(entries)} entries from {path.name}")

    if args.difficulty:
        entries = [e for e in entries if e.difficulty == args.difficulty]
        print(f"Filtered to {len(entries)} entries  difficulty={args.difficulty}")

    if not entries:
        print("No entries to evaluate.")
        sys.exit(0)

    results: list[EvalResult] = []
    for i, entry in enumerate(entries, 1):
        label = entry.query[:55] + ("..." if len(entry.query) > 55 else "")
        print(f"  [{i:02d}/{len(entries)}] {label}", end="", flush=True)
        result = await evaluate_entry(
            entry,
            api_url=args.api_url,
            ollama_url=args.ollama_url,
            llm_model=args.model,
            k=args.top_k,
            run_judge=args.judge,
        )
        if result.error:
            print(f"  ERR: {result.error[:50]}")
        else:
            judge_str = f"  judge={result.llm_score:.0f}/5" if result.llm_score else ""
            print(f"  P@{args.top_k}={result.retrieval.precision_at_k:.2f}  MRR={result.retrieval.mrr:.2f}{judge_str}")
        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.evaluator",
        description="Evaluate vigilant-rag retrieval and generation quality.",
    )
    parser.add_argument(
        "--dataset", default="data/eval/golden_dataset.jsonl",
        help="Path to golden dataset JSONL (default: data/eval/golden_dataset.jsonl)",
    )
    parser.add_argument("--api-url", default="http://localhost:8080")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--judge", action="store_true",
        help="Run LLM-as-judge on each answer (adds ~10-30s per query)",
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard"],
        help="Evaluate only entries of this difficulty",
    )
    parser.add_argument("--output", help="Write full results JSON to this path")
    args = parser.parse_args()

    results = asyncio.run(_run(args))
    print_report(results, args.top_k)

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"Results written to {out}")


if __name__ == "__main__":
    main()
