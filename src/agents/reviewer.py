"""Reviewer agent — checks quality, suggests improvements, and validates output."""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.agents.base import BaseAgent
from src.llm import LLMClient
from src.memory.models import Subtask


class Reviewer(BaseAgent):
    """Agent that reviews output for quality, accuracy, and completeness."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        super().__init__("reviewer", llm_client)

    def execute(self, subtask: Subtask, context: Optional[Dict[str, Any]] = None) -> str:
        """Review content and provide quality assessment."""
        content_to_review = subtask.input_data or ""
        description = subtask.description

        review = self.llm.chat_json(
            messages=[{"role": "user", "content": f"Review the following content based on the original task. Return a JSON object with 'score' (1-10), 'feedback' (string), and 'suggestions' (array of strings).\n\nTask: {description}\n\nContent:\n{content_to_review[:10000]}\n\nEvaluate on: accuracy, completeness, clarity, structure, and factual correctness."}],
            system_prompt="You are a quality assurance reviewer. Evaluate content critically and provide actionable feedback. Be constructive and specific.",
        )

        score = review.get("score", 5)
        feedback = review.get("feedback", "No specific feedback.")
        suggestions = review.get("suggestions", [])
        is_approved = score >= 6

        result_parts = [
            f"## Review Results",
            f"**Score:** {score}/10",
            f"**Status:** {'APPROVED' if is_approved else 'NEEDS IMPROVEMENT'}",
            f"",
            f"**Feedback:** {feedback}",
        ]

        if suggestions:
            result_parts.append(f"\n**Suggestions for improvement:**")
            for i, s in enumerate(suggestions, 1):
                result_parts.append(f"{i}. {s}")

        if not is_approved:
            # Perform a quality improvement pass
            improved = self.llm.chat(
                messages=[{"role": "user", "content": f"Original task: {description}\n\nContent that needs improvement:\n{content_to_review[:10000]}\n\nReview feedback: {feedback}\nSuggestions: {', '.join(suggestions[:3])}\n\nPlease rewrite this content addressing the feedback above."}],
                system_prompt="You are an editor improving content based on review feedback. Maintain factual accuracy while enhancing quality.",
            )
            result_parts.append(f"\n**Improved version:**\n\n{improved}")
            return "\n".join(result_parts)

        return "\n".join(result_parts)
