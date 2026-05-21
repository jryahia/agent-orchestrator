"""Deliverer agent — sends final output via file, Telegram, or email."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from src.agents.base import BaseAgent
from src.llm import LLMClient
from src.memory.models import Subtask


class Deliverer(BaseAgent):
    """Agent that delivers the final result to the user."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        super().__init__("deliverer", llm_client)

    def execute(self, subtask: Subtask, context: Optional[Dict[str, Any]] = None) -> str:
        """Deliver the result based on the delivery method."""
        content = subtask.input_data or ""
        description = subtask.description

        # Parse delivery method from description or metadata
        delivery_method = subtask.metadata.get("delivery_method", "file")
        filename = subtask.metadata.get("filename", "output.md")

        if delivery_method == "file":
            return self._deliver_file(content, filename)
        elif delivery_method == "telegram":
            return self._deliver_telegram(content)
        elif delivery_method == "email":
            return self._deliver_email(content)
        else:
            return self._deliver_file(content, filename)

    def _deliver_file(self, content: str, filename: str) -> str:
        """Save content to a file."""
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Result delivered to file: {filepath}"

    def _deliver_telegram(self, content: str) -> str:
        """Send via Telegram bot (placeholder — needs bot token and chat ID)."""
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not bot_token or not chat_id:
            return "Telegram delivery not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
        try:
            import httpx
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            resp = httpx.post(url, json={"chat_id": chat_id, "text": content[:4096]})
            resp.raise_for_status()
            return "Result delivered via Telegram."
        except Exception as e:
            return f"Telegram delivery failed: {e}"

    def _deliver_email(self, content: str) -> str:
        """Send via email (placeholder — needs SMTP config)."""
        return "Email delivery requires SMTP configuration. Use --output file instead or set EMAIL_* environment variables."
