"""Deliverer agent — sends final output via file, Telegram, email, or WhatsApp."""
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

        # Also check description for delivery hints
        desc_lower = (description or "").lower()
        if "whatsapp" in desc_lower:
            delivery_method = "whatsapp"
        if "telegram" in desc_lower:
            delivery_method = "telegram"
        if "email" in desc_lower:
            delivery_method = "email"

        if delivery_method == "file":
            return self._deliver_file(content, filename)
        elif delivery_method == "telegram":
            return self._deliver_telegram(content)
        elif delivery_method == "email":
            return self._deliver_email(content)
        elif delivery_method == "whatsapp":
            return self._deliver_whatsapp(content)
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

    def _deliver_whatsapp(self, content: str) -> str:
        """Send via WhatsApp Business API."""
        token = os.environ.get("WHATSAPP_TOKEN", "")
        phone_number_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
        if not token or not phone_number_id:
            return "WhatsApp not configured. Set WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID env vars."
        try:
            import httpx
            url = f"https://graph.facebook.com/v21.0/{phone_number_id}/messages"
            resp = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": os.environ.get("WHATSAPP_TO", ""),
                    "type": "text",
                    "text": {"preview_url": False, "body": content[:4096]},
                },
            )
            resp.raise_for_status()
            return "Result delivered via WhatsApp."
        except Exception as e:
            return f"WhatsApp delivery failed: {e}"
