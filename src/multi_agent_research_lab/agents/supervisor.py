"""Supervisor / router skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

# Routing destinations the supervisor can emit
ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing policy (in priority order):
    1. If iteration >= max_iterations  → done (hard cap).
    2. If research_notes is missing    → researcher.
    3. If analysis_notes is missing    → analyst.
    4. If final_answer is missing      → writer.
    5. Otherwise                       → done.
    """

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Inspect shared state and append the next route to route_history."""

        settings = get_settings()

        # ── Hard stop ──────────────────────────────────────────────────────
        if state.iteration >= settings.max_iterations:
            logger.warning(
                "Supervisor: max_iterations=%d reached – forcing done.",
                settings.max_iterations,
            )
            state.record_route(ROUTE_DONE)
            return state

        # ── State-based routing ────────────────────────────────────────────
        if not state.research_notes:
            next_route = ROUTE_RESEARCHER
        elif not state.analysis_notes:
            next_route = ROUTE_ANALYST
        elif not state.final_answer:
            next_route = ROUTE_WRITER
        else:
            next_route = ROUTE_DONE

        logger.info(
            "Supervisor: iter=%d → %s  (research=%s, analysis=%s, answer=%s)",
            state.iteration,
            next_route,
            bool(state.research_notes),
            bool(state.analysis_notes),
            bool(state.final_answer),
        )

        state.record_route(next_route)
        return state
