"""Planner — breaks a goal into subtasks using the LLM."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.llm import LLMClient
from src.memory.models import Subtask, SubtaskStatus


class Planner:
    """Breaks down a high-level goal into manageable subtasks."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def plan(
        self,
        goal: str,
        agents: List[str],
        workflow_id: str,
        max_depth: int = 3,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Subtask]:
        """Decompose a goal into a list of subtasks."""
        agent_descriptions = {
            "researcher": "Searches the web and extracts relevant information. Use for fact-finding, data gathering, and research tasks.",
            "writer": "Summarizes, formats, and structures content into polished output. Use for writing reports, summaries, and formatted documents.",
            "reviewer": "Reviews content for quality, accuracy, and completeness. Use for quality assurance and improvement suggestions.",
            "deliverer": "Delivers the final result via file, Telegram, or email. Use for output delivery.",
        }

        available = {a: agent_descriptions.get(a, a) for a in agents}
        system_prompt = """You are a workflow planner. Decompose a user's goal into a sequence of subtasks.
For each subtask, specify:
- description: what to do (clear and actionable)
- agent_type: which agent handles it (researcher, writer, reviewer, deliverer)
- depth: nesting level (0 for top-level)

Rules:
1. Each subtask must be assigned to an available agent type
2. Keep descriptions focused and specific
3. Order subtasks logically (research before write, write before review)
4. Max depth is 3 levels
5. Return as JSON with a "tasks" array"""

        prompt = f"""Goal: {goal}

Available agents and their capabilities:
{json.dumps(available, indent=2)}

Create a plan with subtasks that accomplishes this goal. Return JSON format with 'tasks' array."""

        try:
            result = self.llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=system_prompt,
                temperature=0.3,
            )
        except Exception as e:
            # Fallback: generate a simple plan
            result = self._fallback_plan(goal, agents)

        tasks_data = result.get("tasks", [])
        if not tasks_data:
            # If LLM returns empty, use fallback
            tasks_data = self._fallback_plan(goal, agents).get("tasks", [])

        subtasks = []
        for i, task_data in enumerate(tasks_data):
            agent_type = task_data.get("agent_type", "").lower()
            if agent_type not in agents:
                # Map to available agent
                if "research" in task_data.get("description", "").lower() and "researcher" in agents:
                    agent_type = "researcher"
                elif "write" in task_data.get("description", "").lower() or "summar" in task_data.get("description", "").lower():
                    agent_type = "writer" if "writer" in agents else (agents[0] if agents else "researcher")
                else:
                    agent_type = agents[0] if agents else "researcher"

            depth = min(task_data.get("depth", 0), max_depth)
            subtask = Subtask(
                workflow_id=workflow_id,
                description=task_data.get("description", f"Task {i+1}"),
                agent_type=agent_type,
                status=SubtaskStatus.PENDING,
                depth=depth,
                max_depth=max_depth,
                input_data=task_data.get("input_data"),
                metadata={"order": i, "goal": goal},
            )
            subtasks.append(subtask)

        return subtasks

    def _fallback_plan(self, goal: str, agents: List[str]) -> Dict[str, Any]:
        """Generate a simple fallback plan when LLM is not available."""
        tasks = []
        step = 1

        if "researcher" in agents:
            tasks.append({
                "description": f"Research: {goal}",
                "agent_type": "researcher",
                "depth": 0,
            })
            step += 1

        if "writer" in agents:
            tasks.append({
                "description": f"Write summary report for: {goal}",
                "agent_type": "writer",
                "depth": 0,
            })
            step += 1

        if "reviewer" in agents:
            tasks.append({
                "description": f"Review and improve the output for: {goal}",
                "agent_type": "reviewer",
                "depth": 0,
            })

        if "deliverer" in agents:
            tasks.append({
                "description": f"Deliver the final result for: {goal}",
                "agent_type": "deliverer",
                "depth": 0,
            })

        return {"tasks": tasks}
