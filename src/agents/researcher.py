"""Researcher agent — performs web search and content extraction."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from urllib.parse import quote
from bs4 import BeautifulSoup

from src.agents.base import BaseAgent
from src.llm import LLMClient
from src.memory.models import Subtask


class Researcher(BaseAgent):
    """Agent that searches the web and extracts content from pages."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        super().__init__("researcher", llm_client)
        self.http_client = httpx.Client(timeout=30.0, follow_redirects=True)

    def execute(self, subtask: Subtask, context: Optional[Dict[str, Any]] = None) -> str:
        """Execute research: search, extract, and synthesize findings."""
        query = subtask.input_data or subtask.description
        context_info = ""
        if context and "previous_results" in context:
            context_info = f"\nPrevious context:\n{context['previous_results']}"

        # Dry-run: skip real web searches
        if self.llm._dry_run:
            return f"[DRY RUN] Research would search: {query}"

        # Use LLM to decide search strategy and interpret results
        search_plan = self.llm.chat_json(
            messages=[{"role": "user", "content": f"Research query: {query}\n{context_info}\nCreate a search plan with 2-3 search queries to answer this comprehensively. Return a single JSON object with a 'queries' array of strings."}],
            system_prompt="You are a research strategist. Plan web searches to gather comprehensive information.",
        )

        queries = search_plan.get("queries", [query])
        all_content = []

        for search_query in queries[:3]:  # Max 3 searches
            try:
                content = self._search_and_extract(search_query)
                all_content.append(f"--- Results for: {search_query} ---\n{content}")
            except Exception as e:
                all_content.append(f"--- Failed search: {search_query} ---\nError: {e}")

        combined = "\n\n".join(all_content)

        # Synthesize with LLM
        synthesis = self.llm.chat(
            messages=[{"role": "user", "content": f"Research query: {query}\n\nRaw research data:\n{combined[:8000]}\n\nSynthesize the findings into a coherent, well-structured research report."}],
            system_prompt="You are a research analyst. Synthesize raw search results into a clear, comprehensive research report with key findings, sources, and analysis.",
        )

        return synthesis

    def _search_and_extract(self, query: str) -> str:
        """Search via a simple approach and extract page content."""
        # Use a search API or scrape approach
        search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        try:
            resp = self.http_client.get(search_url, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract result links
            results = []
            for result in soup.select(".result")[:5]:
                link_el = result.select_one("a.result__a")
                snippet_el = result.select_one(".result__snippet")
                if link_el:
                    title = link_el.get_text(strip=True)
                    href = link_el.get("href", "")
                    snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                    results.append(f"Title: {title}\nURL: {href}\nSnippet: {snippet}")

            if results:
                return "\n\n".join(results)

            # Fallback: try to extract meaningful text
            text = soup.get_text(separator="\n", strip=True)
            lines = [l for l in text.split("\n") if len(l) > 40][:20]
            return "\n".join(lines) if lines else f"No content extracted for: {query}"

        except httpx.HTTPError as e:
            return f"Search failed: {e}"
        except Exception as e:
            return f"Extraction error: {e}"

    def _extract_page_content(self, url: str) -> str:
        """Extract readable content from a specific URL."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        try:
            resp = self.http_client.get(url, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove non-content elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            lines = [l for l in text.split("\n") if len(l) > 30][:50]
            return "\n".join(lines)
        except Exception as e:
            return f"Failed to extract content from {url}: {e}"
