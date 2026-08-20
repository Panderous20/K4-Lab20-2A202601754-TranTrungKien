"""LangGraph workflow — Supervisor → Researcher → Analyst → Writer."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

# ── Node name constants ────────────────────────────────────────────────────────
NODE_SUPERVISOR = "supervisor"
NODE_RESEARCHER = "researcher"
NODE_ANALYST = "analyst"
NODE_WRITER = "writer"


def _state_to_dict(state: ResearchState) -> dict[str, Any]:
    """Convert Pydantic model to plain dict for LangGraph."""
    return state.model_dump()


def _dict_to_state(data: dict[str, Any]) -> ResearchState:
    """Reconstruct ResearchState from the dict LangGraph returns."""
    return ResearchState.model_validate(data)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent LangGraph graph.

    Graph topology:
        supervisor ──► researcher ──► supervisor
                  ──► analyst    ──► supervisor
                  ──► writer     ──► supervisor
                  ──► END  (when route == 'done')
    """

    def build(self) -> Any:
        """Create and return a compiled LangGraph."""

        # Wrap each agent so it works on plain dicts (LangGraph state)
        supervisor = SupervisorAgent()
        researcher = ResearcherAgent()
        analyst = AnalystAgent()
        writer = WriterAgent()

        def _supervisor_node(data: dict[str, Any]) -> dict[str, Any]:
            with trace_span("supervisor", {"iteration": data.get("iteration", 0)}) as span:
                state = _dict_to_state(data)
                state = supervisor.run(state)
                span["next_route"] = state.route_history[-1] if state.route_history else None
                return _state_to_dict(state)

        def _researcher_node(data: dict[str, Any]) -> dict[str, Any]:
            req = data.get("request") or {}
            query = req.get("query") if isinstance(req, dict) else getattr(req, "query", "")
            with trace_span("researcher", {"query": query}) as span:
                state = _dict_to_state(data)
                state = researcher.run(state)
                span["notes_length"] = len(state.research_notes or "")
                return _state_to_dict(state)

        def _analyst_node(data: dict[str, Any]) -> dict[str, Any]:
            with trace_span(
                "analyst", {"notes_length": len(data.get("research_notes") or "")}
            ) as span:
                state = _dict_to_state(data)
                state = analyst.run(state)
                span["analysis_length"] = len(state.analysis_notes or "")
                return _state_to_dict(state)

        def _writer_node(data: dict[str, Any]) -> dict[str, Any]:
            with trace_span(
                "writer", {"analysis_length": len(data.get("analysis_notes") or "")}
            ) as span:
                state = _dict_to_state(data)
                state = writer.run(state)
                span["answer_length"] = len(state.final_answer or "")
                return _state_to_dict(state)

        # ── Conditional router ─────────────────────────────────────────────
        def _route_decision(data: dict[str, Any]) -> str:
            """Read the last entry in route_history to decide the next node."""
            route_history: list[str] = data.get("route_history", [])
            last_route = route_history[-1] if route_history else ROUTE_DONE
            logger.debug("Graph router: last_route=%s", last_route)
            if last_route == ROUTE_RESEARCHER:
                return NODE_RESEARCHER
            if last_route == ROUTE_ANALYST:
                return NODE_ANALYST
            if last_route == ROUTE_WRITER:
                return NODE_WRITER
            return END  # ROUTE_DONE or unknown

        # ── Build graph ────────────────────────────────────────────────────
        graph: StateGraph = StateGraph(dict)

        graph.add_node(NODE_SUPERVISOR, _supervisor_node)
        graph.add_node(NODE_RESEARCHER, _researcher_node)
        graph.add_node(NODE_ANALYST, _analyst_node)
        graph.add_node(NODE_WRITER, _writer_node)

        # Entry point
        graph.set_entry_point(NODE_SUPERVISOR)

        # Conditional edges out of supervisor
        graph.add_conditional_edges(
            NODE_SUPERVISOR,
            _route_decision,
            {
                NODE_RESEARCHER: NODE_RESEARCHER,
                NODE_ANALYST: NODE_ANALYST,
                NODE_WRITER: NODE_WRITER,
                END: END,
            },
        )

        # After each worker → back to supervisor for next routing decision
        graph.add_edge(NODE_RESEARCHER, NODE_SUPERVISOR)
        graph.add_edge(NODE_ANALYST, NODE_SUPERVISOR)
        graph.add_edge(NODE_WRITER, NODE_SUPERVISOR)

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Compile the graph, invoke it with initial state, return final ResearchState."""

        compiled = self.build()
        logger.info("MultiAgentWorkflow: starting run for query=%r", state.request.query)

        with trace_span("multi_agent_workflow", {"query": state.request.query}) as span:
            result_dict: dict[str, Any] = compiled.invoke(_state_to_dict(state))
            final_state = _dict_to_state(result_dict)
            span["iterations"] = final_state.iteration
            span["route_history"] = final_state.route_history
            span["has_answer"] = bool(final_state.final_answer)

        logger.info(
            "MultiAgentWorkflow: finished  route_history=%s  iterations=%d",
            final_state.route_history,
            final_state.iteration,
        )
        return final_state
