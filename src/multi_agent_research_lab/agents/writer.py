"""Writer agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer` by synthesising all notes via LLM."""

        logger.info("WriterAgent: synthesising final answer for %r", state.request.query)

        client = LLMClient()
        system_prompt = (
            "You are an expert technical writer. Given a research query, research notes, "
            "and analyst insights, write a clear, well-structured final report:\n"
            "- Start with an executive summary (2-3 sentences).\n"
            "- Cover main findings with supporting evidence.\n"
            "- Include a 'References' section citing the sources.\n"
            "- End with a 'Conclusion' paragraph.\n"
            "Audience: technical learners. Format: markdown."
        )
        sources_text = "\n".join(
            f"- {s.title}: {s.snippet}" for s in state.sources
        ) if state.sources else "No external sources."

        response = client.complete(
            system_prompt=system_prompt,
            user_prompt=(
                f"Query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                f"Analysis:\n{state.analysis_notes}\n\n"
                f"Sources:\n{sources_text}"
            ),
        )

        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
            )
        )
        state.add_trace_event("writer_done", {"answer_length": len(response.content)})
        logger.info("WriterAgent: done, final_answer=%d chars", len(response.content))
        return state

