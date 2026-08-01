"""CustomAgent — a user-defined agent with a custom name and system prompt."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.agents.base import BaseAgent
from src.llm import LLMClient
from src.memory.models import Subtask


class CustomAgent(BaseAgent):
    """A user-defined agent with a configurable name and system prompt.

    This allows users to create their own agents from the TUI without
    writing any code. The agent simply calls the LLM with the custom
    system prompt, passing the subtask description and any context.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm_client: Optional[LLMClient] = None,
    ):
        super().__init__(name=name, llm_client=llm_client)
        self._system_prompt = system_prompt

    def get_system_prompt(self) -> str:
        """Return the custom system prompt the user provided."""
        return self._system_prompt

    def execute(self, subtask: Subtask, context: Optional[Dict[str, Any]] = None) -> str:
        """Execute a subtask by calling the LLM with the custom system prompt.

        Args:
            subtask: The subtask describing what this agent should do.
            context: Optional shared context (previous results, etc.).

        Returns:
            The LLM response as a string.
        """
        if context is None:
            context = {}

        previous = context.get("previous_results", "")

        user_message_parts = [f"Task: {subtask.description}"]
        if subtask.input_data:
            user_message_parts.append(f"\nInput: {subtask.input_data}")
        if previous:
            user_message_parts.append(f"\nPrevious results:\n{previous}")

        user_message = "\n".join(user_message_parts)

        response = self.llm.chat(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=self._system_prompt,
            temperature=0.7,
        )

        return response or f"Custom agent '{self.name}' completed: {subtask.description}"
