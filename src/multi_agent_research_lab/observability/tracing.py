"""Tracing hooks — LangSmith primary, JSON-file fallback.

Provider priority:
1. LangSmith  (when LANGSMITH_API_KEY is set in .env)
2. Local JSON  (always written to reports/traces/ as a side-channel log)

Usage in agent code:
    from multi_agent_research_lab.observability.tracing import trace_span

    with trace_span("researcher", {"query": q}) as span:
        ...
        span["output_tokens"] = 123
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

# ── LangSmith client (optional) ───────────────────────────────────────────────
_ls_client: Any = None


def _get_langsmith_client() -> Any:
    """Return a cached LangSmith Client, or None if not configured."""
    global _ls_client
    if _ls_client is not None:
        return _ls_client

    api_key = os.getenv("LANGSMITH_API_KEY", "")
    if not api_key:
        return None

    try:
        from langsmith import Client  # type: ignore[import]

        _ls_client = Client(api_key=api_key)
        logger.info("Tracing: LangSmith client initialised (project=%s)", os.getenv("LANGSMITH_PROJECT"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tracing: LangSmith unavailable — %s", exc)
        _ls_client = None

    return _ls_client


# ── Local trace sink ──────────────────────────────────────────────────────────
_TRACE_DIR = Path("reports/traces")


def _write_local_trace(span: dict[str, Any]) -> None:
    """Append span JSON to a daily trace file for offline inspection."""
    try:
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        trace_file = _TRACE_DIR / f"trace_{date_str}.jsonl"
        with trace_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(span) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Tracing: failed to write local trace — %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────
@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager that wraps a unit of work in a trace span.

    - Sends run to LangSmith if LANGSMITH_API_KEY is configured.
    - Always appends to a local JSONL file under reports/traces/.
    - Yields a mutable ``span`` dict so callers can attach output metadata:

        with trace_span("researcher", {"query": q}) as span:
            result = do_work()
            span["output_tokens"] = result.output_tokens
    """
    attrs = attributes or {}
    started = perf_counter()
    ts_start = datetime.now(timezone.utc).isoformat()

    span: dict[str, Any] = {
        "name": name,
        "attributes": attrs,
        "started_at": ts_start,
        "duration_seconds": None,
    }

    # ── LangSmith: create run ─────────────────────────────────────────────
    ls_run_id: str | None = None
    ls_client = _get_langsmith_client()
    if ls_client is not None:
        try:
            import uuid

            project = os.getenv("LANGSMITH_PROJECT", "multi-agent-research-lab")
            ls_run_id = str(uuid.uuid4())
            ls_client.create_run(
                id=ls_run_id,
                name=name,
                run_type="chain",
                inputs=attrs,
                project_name=project,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracing: LangSmith create_run failed — %s", exc)
            ls_run_id = None

    error: Exception | None = None
    try:
        yield span
    except Exception as exc:  # noqa: BLE001
        error = exc
        span["error"] = str(exc)
        raise
    finally:
        duration = perf_counter() - started
        span["duration_seconds"] = duration

        # ── LangSmith: end run ────────────────────────────────────────────
        if ls_client is not None and ls_run_id is not None:
            try:
                outputs = {k: v for k, v in span.items() if k not in ("name", "attributes", "started_at")}
                ls_client.update_run(
                    ls_run_id,
                    outputs=outputs,
                    error=str(error) if error else None,
                    end_time=datetime.now(timezone.utc),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Tracing: LangSmith update_run failed — %s", exc)

        # ── Local sink ────────────────────────────────────────────────────
        _write_local_trace(span)

        logger.debug("trace_span '%s' finished in %.3fs", name, duration)

