"""Unified LLM client supporting OpenAI, DeepSeek, OpenRouter, and compatible APIs."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


class LLMClient:
    """Unified client for LLM interactions via the openai library.

    Passed parameters (api_key, base_url, model) take priority over
    the corresponding environment variables (LLM_API_KEY, LLM_BASE_URL,
    LLM_MODEL), which serve as fallbacks.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "sk-placeholder")
        self.base_url = base_url or os.environ.get(
            "LLM_BASE_URL", "https://api.openai.com/v1"
        )
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self._dry_run = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")

        if self._dry_run:
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion request to the LLM."""
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        if self._dry_run or self.client is None:
            return self._dry_run_response(messages)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Send a chat completion and parse JSON response."""
        result = self.chat(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r"\{.*\}", result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Failed to parse JSON from LLM response: {result[:200]}")

    def _dry_run_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a mock response for dry-run testing."""
        last_msg = messages[-1]["content"] if messages else ""
        if "json" in last_msg.lower() or "plan" in last_msg.lower():
            return json.dumps({
                "tasks": [
                    {
                        "description": f"Research: {last_msg[:60]}",
                        "agent_type": "researcher",
                        "depth": 0,
                    },
                    {
                        "description": f"Write summary: {last_msg[:60]}",
                        "agent_type": "writer",
                        "depth": 0,
                    },
                ]
            })
        return f"[DRY RUN] Mock response for: {last_msg[:100]}..."
