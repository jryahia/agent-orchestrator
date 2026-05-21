"""Writer agent — summarizes and formats content into structured output."""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.agents.base import BaseAgent
from src.llm import LLMClient
from src.memory.models import Subtask


class Writer(BaseAgent):
    """Agent that formats and structures content into polished output."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        super().__init__("writer", llm_client)

    def execute(self, subtask: Subtask, context: Optional[Dict[str, Any]] = None) -> str:
        """Write/formatted output based on the task description and input data."""
        input_content = subtask.input_data or ""
        description = subtask.description

        context_info = ""
        if context and "previous_results" in context:
            context_info = f"\nPrevious context:\n{context['previous_results']}"

        system_prompt = """You are a professional writer and editor. Your role is to:
1. Take raw research data and transform it into polished, well-structured content
2. Use clear headings, bullet points, and paragraphs for readability
3. Maintain factual accuracy while improving presentation
4. Adapt tone and style based on the intended audience
5. Include a brief summary or executive overview at the top"""

        user_prompt = f"""Task: {description}

Raw content to format:
{input_content[:12000]}
{context_info}

Produce a well-formatted, structured output. Use markdown formatting with headers, lists, and emphasis as appropriate."""

        return self.llm.chat(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=4096,
        )

    def format_result(self, result: str, format_type: str = "markdown") -> str:
        """Format the result in the requested output format."""
        if format_type == "json":
            import json
            return json.dumps({"content": result, "format": "json"}, indent=2)
        elif format_type == "text":
            # Strip markdown formatting
            import re
            text = re.sub(r"#{1,6}\s+", "", result)
            text = re.sub(r"\*\*", "", text)
            text = re.sub(r"\*", "", text)
            text = re.sub(r"`{1,3}", "", text)
            text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
            return text.strip()
        return result  # markdown
