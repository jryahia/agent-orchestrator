"""Base agent class from which all specialized agents inherit."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.llm import LLMClient
from src.memory.models import Subtask


class BaseAgent(ABC):
    """Abstract base class for all worker agents."""

    def __init__(self, name: str, llm_client: Optional[LLMClient] = None):
        self.name = name
        self.llm = llm_client or LLMClient()

    @abstractmethod
    def execute(self, subtask: Subtask, context: Optional[Dict[str, Any]] = None) -> str:
        """Execute the agent's task and return the result."""
        pass

    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent type."""
        return f"You are a {self.name} agent in a multi-agent orchestration system."

    def format_result(self, result: str, format_type: str = "text") -> str:
        """Format the result in the requested output format."""
        return result
