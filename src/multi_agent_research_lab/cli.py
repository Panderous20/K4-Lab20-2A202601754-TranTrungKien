"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline with a real LLM call."""

    import time

    from multi_agent_research_lab.observability.tracing import trace_span
    from multi_agent_research_lab.services.llm_client import LLMClient

    _init()
    request = _parse_query(query)

    system_prompt = (
        "You are a research assistant. Given a research query, provide a concise, "
        "well-structured answer with key findings and references where possible."
    )

    client = LLMClient()

    with trace_span("baseline", {"query": request.query}) as span:
        start = time.perf_counter()
        response = client.complete(system_prompt=system_prompt, user_prompt=request.query)
        latency = time.perf_counter() - start

        state = ResearchState(request=request)
        state.final_answer = response.content
        span["input_tokens"] = response.input_tokens
        span["output_tokens"] = response.output_tokens
        span["cost_usd"] = response.cost_usd
        span["latency_seconds"] = latency

    # ── Display answer ──────────────────────────────────────────────
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))

    # ── Display metrics ─────────────────────────────────────────────
    metrics_lines = [
        f"Latency:       {latency:.2f}s",
        f"Input tokens:  {response.input_tokens}",
        f"Output tokens: {response.output_tokens}",
        f"Est. cost:     ${response.cost_usd:.6f}" if response.cost_usd else "Est. cost:     N/A",
    ]
    console.print(Panel.fit("\n".join(metrics_lines), title="Baseline Metrics", style="cyan"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
