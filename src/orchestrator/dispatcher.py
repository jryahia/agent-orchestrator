"""Dispatcher — assigns subtasks to appropriate agents and manages execution."""
from __future__ import annotations

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from src.agents import Deliverer, Researcher, Reviewer, Writer
from src.agents.base import BaseAgent
from src.agents.custom_agent import CustomAgent
from src.custom_agent_store import CustomAgentStore
from src.llm import LLMClient
from src.memory.models import Subtask, SubtaskStatus
from src.memory.task_store import TaskStore
from src.orchestrator.tracker import Tracker


class Dispatcher:
    """Assigns subtasks to agents and orchestrates execution."""

    def __init__(
        self,
        task_store: Optional[TaskStore] = None,
        llm_client: Optional[LLMClient] = None,
        tracker: Optional[Tracker] = None,
        parallel: bool = False,
        custom_agents_dir: Optional[str] = None,
    ):
        self.store = task_store or TaskStore()
        self.llm = llm_client or LLMClient()
        self.tracker = tracker or Tracker(self.store)
        self.parallel = parallel
        self._agents: Dict[str, BaseAgent] = {}
        # Custom agent store — load persisted definitions
        self._custom_agent_store = CustomAgentStore(data_dir=custom_agents_dir)
        self._loaded_custom_agents: Dict[str, Dict[str, str]] = (
            self._custom_agent_store.to_dict()
        )

    def register_custom_agent(self, name: str, system_prompt: str) -> None:
        """Register a new custom agent at runtime and persist it.

        Once registered, the agent is immediately available via _get_agent().
        """
        self._custom_agent_store.add(name, system_prompt)
        # Refresh the loaded agents dict
        self._loaded_custom_agents = self._custom_agent_store.to_dict()
        # Clear any cached agent with this name so it gets re-created
        self._agents.pop(name, None)

    def _get_agent(self, agent_type: str) -> BaseAgent:
        """Get or create an agent by type."""
        if agent_type not in self._agents:
            if agent_type == "researcher":
                self._agents[agent_type] = Researcher(self.llm)
            elif agent_type == "writer":
                self._agents[agent_type] = Writer(self.llm)
            elif agent_type == "reviewer":
                self._agents[agent_type] = Reviewer(self.llm)
            elif agent_type == "deliverer":
                self._agents[agent_type] = Deliverer(self.llm)
            elif agent_type in self._loaded_custom_agents:
                # Create a CustomAgent on-the-fly from the stored definition
                info = self._loaded_custom_agents[agent_type]
                self._agents[agent_type] = CustomAgent(
                    name=agent_type,
                    system_prompt=info["system_prompt"],
                    llm_client=self.llm,
                )
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
        return self._agents[agent_type]

    def execute_workflow(
        self,
        workflow_id: str,
        subtasks: List[Subtask],
    ) -> Dict[str, Any]:
        """Execute all subtasks in a workflow."""
        results = {}
        self.tracker.start_workflow(workflow_id)

        if self.parallel:
            results = self._execute_parallel(subtasks)
        else:
            results = self._execute_sequential(subtasks)

        # Compile final result
        final_output = self._compile_results(subtasks, results)
        self.store.update_workflow_output(workflow_id, final_output)
        self.tracker.complete_workflow(workflow_id)

        return {
            "workflow_id": workflow_id,
            "final_output": final_output,
            "subtask_results": results,
        }

    def _execute_sequential(self, subtasks: List[Subtask]) -> Dict[str, str]:
        """Execute subtasks one by one, passing context forward."""
        results: Dict[str, str] = {}
        context: Dict[str, Any] = {"previous_results": ""}

        for subtask in subtasks:
            result = self._execute_subtask_with_retry(subtask, context)
            results[subtask.id] = result
            context["previous_results"] = (context.get("previous_results", "") + "\n\n" + result).strip()
            # Tracker already saved the subtask status; no need to save again

        return results

    def _execute_parallel(self, subtasks: List[Subtask]) -> Dict[str, str]:
        """Execute independent subtasks in parallel."""
        results: Dict[str, str] = {}
        context: Dict[str, Any] = {"previous_results": ""}

        # Group by dependency level (sequential within same level)
        levels: Dict[int, List[Subtask]] = {}
        for st in subtasks:
            depth = st.metadata.get("order", 0)
            levels.setdefault(depth // 2, []).append(st)

        for level in sorted(levels.keys()):
            level_tasks = levels[level]
            if len(level_tasks) == 1:
                result = self._execute_subtask_with_retry(level_tasks[0], context)
                results[level_tasks[0].id] = result
                context["previous_results"] += "\n\n" + result
            else:
                with ThreadPoolExecutor(max_workers=len(level_tasks)) as executor:
                    futures = {
                        executor.submit(self._execute_subtask_with_retry, st, context): st
                        for st in level_tasks
                    }
                    for future in as_completed(futures):
                        st = futures[future]
                        try:
                            result = future.result()
                            results[st.id] = result
                            context["previous_results"] += "\n\n" + result
                        except Exception as e:
                            results[st.id] = f"Error: {e}"
                            self.tracker.fail_subtask(st.id, str(e))

        return results

    def _execute_subtask_with_retry(self, subtask: Subtask, context: Dict[str, Any]) -> str:
        """Execute a subtask with retry logic."""
        attempt = 0
        max_attempts = 1 + subtask.max_retries  # First attempt + retries
        
        while attempt < max_attempts:
            attempt += 1
            try:
                self.tracker.start_subtask(subtask.id)
                agent = self._get_agent(subtask.agent_type)
                result = agent.execute(subtask, context)
                self.tracker.complete_subtask(subtask.id, result)
                return result
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                subtask.error_message = error_msg
                subtask.increment_retry()
                self.tracker.fail_subtask(subtask.id, error_msg)
                if attempt >= max_attempts:
                    return f"Failed after {attempt} attempt(s): {error_msg}"

        return subtask.error_message or "Subtask execution failed."

    def _compile_results(self, subtasks: List[Subtask], results: Dict[str, str]) -> str:
        """Compile all subtask results into a final output."""
        parts = ["# Multi-Agent Workflow Results\n"]

        for subtask in subtasks:
            # Get latest status from the store
            latest = self.store.get_subtask(subtask.id) or subtask
            latest_status = latest.status

            agent_emoji = {
                "researcher": "🔍",
                "writer": "✍️",
                "reviewer": "✅",
                "deliverer": "📤",
            }.get(subtask.agent_type, "🤖")

            status_icon = "✅" if latest_status == SubtaskStatus.COMPLETED else "❌"
            parts.append(f"## {agent_emoji} {subtask.agent_type.title()} Agent {status_icon}")
            parts.append(f"**Task:** {subtask.description}")
            parts.append(f"**Status:** {latest_status.value}")
            if latest.retry_count > 0:
                parts.append(f"**Retries:** {latest.retry_count}")

            result = results.get(subtask.id, "")
            if result:
                parts.append(f"\n{result}")

            parts.append("\n---\n")

        return "\n".join(parts)
