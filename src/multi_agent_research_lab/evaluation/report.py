"""Benchmark report rendering.

Public API:
    render_markdown_report(metrics)  → markdown string
    save_report(metrics)             → writes reports/benchmark_report.md
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from multi_agent_research_lab.core.schemas import BenchmarkMetrics

logger = logging.getLogger(__name__)

_REPORT_PATH = Path("reports/benchmark_report.md")


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a rich markdown report.

    Sections:
    1. Summary table (all runs × all metrics)
    2. Per-metric winner callout
    3. Trade-off analysis narrative
    4. Failure mode paragraph
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        "# Benchmark Report",
        "",
        f"_Generated: {now}_",
        "",
        "## Summary Table",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality / 10 | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for item in metrics:
        cost = "—" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.5f}"
        quality = "—" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "—" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "—" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| **{item.run_name}** | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    # ── Per-metric winners ────────────────────────────────────────────────────
    lines += ["", "## Per-Metric Analysis", ""]

    def _winner(attr: str, lower_is_better: bool = False) -> str:
        candidates = [(m.run_name, getattr(m, attr)) for m in metrics if getattr(m, attr) is not None]
        if not candidates:
            return "—"
        best = min(candidates, key=lambda x: x[1]) if lower_is_better else max(candidates, key=lambda x: x[1])
        return best[0]

    lines += [
        f"- **Fastest** (lowest latency): `{_winner('latency_seconds', lower_is_better=True)}`",
        f"- **Cheapest** (lowest cost): `{_winner('estimated_cost_usd', lower_is_better=True)}`",
        f"- **Highest quality**: `{_winner('quality_score')}`",
        f"- **Best citation coverage**: `{_winner('citation_coverage')}`",
        f"- **Most reliable** (lowest failure rate): `{_winner('failure_rate', lower_is_better=True)}`",
    ]

    # ── Trade-off narrative ───────────────────────────────────────────────────
    lines += [
        "",
        "## Trade-off Analysis",
        "",
        "Multi-agent pipelines route the query through dedicated **Researcher → Analyst → Writer** "
        "agents, each making an independent LLM call. This produces:",
        "",
        "- **Higher quality & citation coverage** — each agent has a focused prompt, leading to "
        "richer, more structured outputs with explicit citations.",
        "- **Higher latency & cost** — 3–4 sequential LLM calls vs. 1 for the baseline. "
        "Wall-clock time scales roughly linearly with the number of agent turns.",
        "",
        "**When to prefer multi-agent:**",
        "- Tasks requiring deep analysis, structured sections, and traceable citations.",
        "- Audiences expecting a research-grade report over a quick answer.",
        "",
        "**When to prefer single-agent baseline:**",
        "- Time-sensitive use cases (< 10 s budget).",
        "- Simple factual queries where one LLM pass is sufficient.",
        "- Cost-sensitive applications at scale.",
    ]

    # ── Failure mode paragraph ────────────────────────────────────────────────
    lines += [
        "",
        "## Observed Failure Modes",
        "",
        "1. **Hallucinated citations** — researcher LLM fabricates plausible-sounding references "
        "(author, journal, year) that do not exist. Mitigation: integrate a real search tool "
        "(Tavily / Bing) to ground sources in live URLs.",
        "2. **Latency spikes** — sequential LLM calls add up; a 60 s timeout can be hit on slow "
        "models or rate-limited keys. Mitigation: parallelise researcher sub-tasks or add caching.",
        "3. **Token budget overflow** — passing full `research_notes` into the analyst prompt risks "
        "exceeding context limits for very long queries. Mitigation: truncate or chunk notes before "
        "passing downstream.",
        "4. **Routing loop** — if an agent fails to populate its expected field (e.g., "
        "`analysis_notes`), the supervisor will keep re-routing to analyst until `max_iterations`. "
        "Mitigation: add an `errors` check in the routing policy and emit `done` on repeated failure.",
        "",
        "---",
        "_Report produced by `evaluation/report.py`. Re-run with `python scripts/run_benchmark.py`._",
    ]

    return "\n".join(lines) + "\n"


def save_report(metrics: list[BenchmarkMetrics], path: Path = _REPORT_PATH) -> Path:
    """Write the markdown report to disk and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_markdown_report(metrics)
    path.write_text(content, encoding="utf-8")
    logger.info("Benchmark report written to %s (%d bytes)", path, len(content))
    return path

