"""Analyst agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes` from research_notes via LLM."""

        logger.info("AnalystAgent: analysing %d chars of research notes", len(state.research_notes or ""))

        client = LLMClient()
        system_prompt = (
            "You are a critical analyst agent. Given research notes, produce:\n"
            "1. **Key Claims** — 3-5 concise claims supported by the notes.\n"
            "2. **Competing Viewpoints** — any alternative perspectives or disagreements.\n"
            "3. **Weak Evidence / Gaps** — identify unsupported claims or missing evidence.\n"
            "4. **Synthesis** — 2-3 sentences linking findings into a coherent narrative.\n"
            "Format: markdown headings."
        )
        response = client.complete(
            system_prompt=system_prompt,
            user_prompt=(
                f"Original query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}"
            ),
        )

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
            )
        )
        state.add_trace_event("analyst_done", {"analysis_length": len(response.content)})
        logger.info("AnalystAgent: done, analysis=%d chars", len(response.content))
        return state

