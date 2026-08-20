"""Benchmark runner — single-agent vs multi-agent comparison.

Metrics collected per run:
- latency_seconds  : wall-clock time for the full runner call
- estimated_cost_usd : sum of per-agent token costs from agent_results
- quality_score    : heuristic 0-10 (output length + citation density)
- citation_coverage: fraction of sources[].title mentioned in final_answer
- failure_rate     : fraction of queries that raised an exception
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _estimate_total_cost(state: ResearchState) -> float | None:
    """Sum cost_usd from sources metadata (set by each agent via LLMClient)."""
    total = 0.0
    found = False
    for src in state.sources:
        cost = src.metadata.get("cost_usd")
        if cost is not None:
            total += float(cost)
            found = True
    # Also scan agent_results for any token counts not captured in sources
    # (analyst / writer don't write to sources)
    PRICE_PER_1K_IN = 0.000150   # gpt-4o-mini defaults
    PRICE_PER_1K_OUT = 0.000600
    for ar in state.agent_results:
        in_tok = ar.metadata.get("input_tokens")
        out_tok = ar.metadata.get("output_tokens")
        if in_tok is not None and out_tok is not None:
            total += (in_tok / 1_000) * PRICE_PER_1K_IN + (out_tok / 1_000) * PRICE_PER_1K_OUT
            found = True
    return round(total, 6) if found else None


def _quality_score(state: ResearchState) -> float:
    """Heuristic quality score 0-10.

    Components:
    - final_answer exists            → base 3 pts
    - final_answer length ≥ 500     → +2 pts
    - final_answer length ≥ 1500    → +1 pt  (up to 3 pts total for length)
    - citation pattern found (Author, YYYY) → +2 pts
    - has markdown headings (##)    → +1 pt
    - research_notes + analysis_notes both present → +1 pt
    """
    if not state.final_answer:
        return 0.0

    score = 3.0  # base: answer exists
    length = len(state.final_answer)
    if length >= 500:
        score += 2.0
    if length >= 1500:
        score += 1.0

    # Citation pattern: (Author, YYYY) or [Author YYYY] or numbered refs
    citation_re = re.compile(r"(\([A-Z][a-z]+.*?\d{4}\)|\[\d+\]|^\d+\.\s+[A-Z])", re.MULTILINE)
    if citation_re.search(state.final_answer):
        score += 2.0

    if "##" in state.final_answer:
        score += 1.0

    if state.research_notes and state.analysis_notes:
        score += 1.0

    return min(score, 10.0)


def _citation_coverage(state: ResearchState) -> float:
    """Fraction of source titles (keywords) found in final_answer."""
    if not state.sources or not state.final_answer:
        return 0.0

    answer_lower = state.final_answer.lower()
    hits = 0
    for src in state.sources:
        # Use first significant word of source title as proxy
        keywords = [w for w in src.title.split() if len(w) > 4]
        if any(kw.lower() in answer_lower for kw in keywords):
            hits += 1
    return round(hits / len(state.sources), 3)


# ── Public API ─────────────────────────────────────────────────────────────────

def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run a single query through ``runner`` and return state + metrics."""

    with trace_span(f"benchmark:{run_name}", {"query": query}) as span:
        started = perf_counter()
        state = runner(query)
        latency = perf_counter() - started

        cost = _estimate_total_cost(state)
        quality = _quality_score(state)
        citation = _citation_coverage(state)

        span["latency_seconds"] = latency
        span["quality_score"] = quality
        span["citation_coverage"] = citation
        span["cost_usd"] = cost

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality,
        citation_coverage=citation,
        failure_rate=0.0,
        notes=f"iter={state.iteration} route={state.route_history}",
    )
    logger.info(
        "Benchmark '%s': latency=%.2fs cost=$%s quality=%.1f citation=%.0f%%",
        run_name, latency, f"{cost:.5f}" if cost else "N/A", quality, (citation or 0) * 100,
    )
    return state, metrics


def run_benchmark_suite(
    run_name: str,
    queries: list[str],
    runner: Runner,
) -> BenchmarkMetrics:
    """Run a list of queries, aggregate metrics, and return a single BenchmarkMetrics.

    failure_rate = fraction of queries that raised an exception.
    Other metrics are averaged across successful runs.
    """
    latencies: list[float] = []
    costs: list[float] = []
    qualities: list[float] = []
    citations: list[float] = []
    failures = 0

    for query in queries:
        try:
            _, m = run_benchmark(f"{run_name}:{query[:30]}", query, runner)
            latencies.append(m.latency_seconds)
            if m.estimated_cost_usd is not None:
                costs.append(m.estimated_cost_usd)
            if m.quality_score is not None:
                qualities.append(m.quality_score)
            if m.citation_coverage is not None:
                citations.append(m.citation_coverage)
        except Exception as exc:  # noqa: BLE001
            logger.error("Benchmark suite '%s' query failed: %s", run_name, exc)
            failures += 1

    n = len(queries)
    failure_rate = failures / n if n else 1.0

    return BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=sum(latencies) / len(latencies) if latencies else 0.0,
        estimated_cost_usd=sum(costs) / len(costs) if costs else None,
        quality_score=sum(qualities) / len(qualities) if qualities else None,
        citation_coverage=sum(citations) / len(citations) if citations else None,
        failure_rate=failure_rate,
        notes=f"n={n} failures={failures}",
    )

