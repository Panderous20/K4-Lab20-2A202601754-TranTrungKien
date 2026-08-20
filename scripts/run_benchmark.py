"""End-to-end benchmark script.

Usage:
    python scripts/run_benchmark.py

Runs a shared set of queries through both the single-agent baseline and the
multi-agent workflow, measures all metrics, prints results, and writes
reports/benchmark_report.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark_suite
from multi_agent_research_lab.evaluation.report import save_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

# ── Setup ──────────────────────────────────────────────────────────────────────
settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

QUERIES = [
    "Research GraphRAG state-of-the-art",
    "What are the main challenges in multi-agent LLM systems?",
]

SYSTEM_PROMPT = (
    "You are a research assistant. Given a research query, provide a concise, "
    "well-structured answer with key findings and references where possible."
)


# ── Runners ────────────────────────────────────────────────────────────────────

def baseline_runner(query: str) -> ResearchState:
    """Single-agent baseline: one LLM call."""
    client = LLMClient()
    response = client.complete(system_prompt=SYSTEM_PROMPT, user_prompt=query)
    state = ResearchState(request=ResearchQuery(query=query))
    state.final_answer = response.content
    # Record cost in sources so benchmark can pick it up
    from multi_agent_research_lab.core.schemas import SourceDocument
    state.sources = [
        SourceDocument(
            title=f"Baseline — {query}",
            url=None,
            snippet=response.content[:200],
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
    ]
    state.record_route("done")
    return state


def multi_agent_runner(query: str) -> ResearchState:
    """Full multi-agent pipeline."""
    workflow = MultiAgentWorkflow()
    return workflow.run(ResearchState(request=ResearchQuery(query=query)))


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  Multi-Agent Research Lab — Benchmark Runner")
    print(f"  Queries: {len(QUERIES)}")
    print(f"{'='*60}\n")

    print("▶ Running BASELINE suite …")
    baseline_metrics = run_benchmark_suite(
        run_name="baseline",
        queries=QUERIES,
        runner=baseline_runner,
    )

    print("\n▶ Running MULTI-AGENT suite …")
    multi_metrics = run_benchmark_suite(
        run_name="multi-agent",
        queries=QUERIES,
        runner=multi_agent_runner,
    )

    all_metrics = [baseline_metrics, multi_metrics]

    report_path = save_report(all_metrics)
    print(f"\n✅ Report saved → {report_path.resolve()}")

    # Print summary table to console
    print("\n--- Summary ---")
    for m in all_metrics:
        cost = f"${m.estimated_cost_usd:.5f}" if m.estimated_cost_usd else "N/A"
        quality = f"{m.quality_score:.1f}/10" if m.quality_score else "N/A"
        citation = f"{m.citation_coverage:.0%}" if m.citation_coverage else "N/A"
        print(
            f"  {m.run_name:<14} | latency={m.latency_seconds:.2f}s "
            f"| cost={cost} | quality={quality} | citation={citation} "
            f"| failure={m.failure_rate:.0%}"
        )
