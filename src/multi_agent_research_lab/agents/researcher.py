"""Researcher agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes` via LLM."""

        logger.info("ResearcherAgent: gathering research for query=%r", state.request.query)

        client = LLMClient()
        system_prompt = (
            "You are a meticulous research agent. Given a research query, produce:\n"
            "1. A numbered list of key findings (5-8 bullet points).\n"
            "2. A 'Sources' section with 3-5 cited references (author, title, year if known).\n"
            "Be factual and concise. Format: plain text."
        )
        response = client.complete(
            system_prompt=system_prompt,
            user_prompt=f"Research query: {state.request.query}",
        )

        state.research_notes = response.content

        # Represent the LLM-generated content as a single synthetic source document
        state.sources = [
            SourceDocument(
                title=f"LLM Research Notes — {state.request.query}",
                url=None,
                snippet=response.content[:300],
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        ]

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
            )
        )
        state.add_trace_event("researcher_done", {"notes_length": len(response.content)})
        logger.info("ResearcherAgent: done, notes=%d chars", len(response.content))
        return state

