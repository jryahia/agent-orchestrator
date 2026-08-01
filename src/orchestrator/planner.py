"""Planner — breaks a goal into subtasks using the LLM."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.llm import LLMClient
from src.memory.models import Subtask, SubtaskStatus


# Agent descriptions shared across methods
AGENT_DESCRIPTIONS = {
    "researcher": "Searches the web and extracts relevant information. Use for fact-finding, data gathering, and research tasks.",
    "writer": "Summarizes, formats, and structures content into polished output. Use for writing reports, summaries, and formatted documents.",
    "reviewer": "Reviews content for quality, accuracy, and completeness. Use for quality assurance and improvement suggestions.",
    "deliverer": "Delivers the final result via file, Telegram, or email. Use for output delivery.",
}

# Built-in agents (used to distinguish from custom agents in fallback)
BUILTIN_AGENTS = {"researcher", "writer", "reviewer", "deliverer"}


class Planner:
    """Breaks down a high-level goal into manageable subtasks."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def _build_available_descriptions(self, agents: List[str]) -> Dict[str, str]:
        """Build a dict of agent -> description, handling custom agents gracefully."""
        available = {}
        for a in agents:
            if a in AGENT_DESCRIPTIONS:
                available[a] = AGENT_DESCRIPTIONS[a]
            else:
                # Custom agent — use a generic description based on its name
                available[a] = (
                    f"Performs a specialized task as '{a}'. "
                    f"Use this agent for tasks that fit the '{a}' role."
                )
        return available

    def plan(
        self,
        goal: str,
        agents: List[str],
        workflow_id: str,
        max_depth: int = 3,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Subtask]:
        """Decompose a goal into a list of subtasks."""
        available = self._build_available_descriptions(agents)
        system_prompt = """You are a workflow planner. Decompose a user's goal into a sequence of subtasks.
For each subtask, specify:
- description: what to do (clear and actionable)
- agent_type: which agent handles it (researcher, writer, reviewer, deliverer, or a custom agent name)
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

Create a plan with subtasks that accomplishes this goal. Return a JSON object with a 'tasks' array where each task has 'description', 'agent_type', and 'depth' fields."""

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

    def plan_from_skill(
        self,
        skill_name: str,
        prompt_text: str,
        agent_type: str,
        agents: List[str],
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Subtask]:
        """Generate a plan centered on a saved skill's prompt for a specific agent.

        Creates a single subtask targeting the skill's agent_type with the skill's
        prompt_text as the description, then wraps it in a proper full plan
        (research → skill execution → write/review/deliver as needed).
        """
        # Build a plan with the skill subtask at its core
        skill_desc = f"[SKILL: {skill_name}] {prompt_text}"

        # Ensure the skill's agent type is in the available agents list
        available_agents = agents[:]
        if agent_type not in available_agents:
            available_agents.append(agent_type)

        # Use the LLM to generate a plan around the skill prompt
        available = self._build_available_descriptions(available_agents)
        system_prompt = """You are a workflow planner. A skill has been selected — a reusable
prompt targeting a specific agent. Build a complete workflow plan that includes
the skill execution as the core subtask, with appropriate supporting subtasks
(research before, write/review after) based on the skill's agent type.

For each subtask, specify:
- description: what to do (clear and actionable)
- agent_type: which agent handles it (researcher, writer, reviewer, deliverer)
- depth: nesting level (0 for top-level)

Rules:
1. One subtask MUST be the skill execution with the exact description provided
2. Each subtask must be assigned to an available agent type
3. Order subtasks logically (research before skill, skill before write/review)
4. Max depth is 3 levels
5. Return as JSON with a "tasks" array"""

        prompt = f"""Skill Name: {skill_name}
Skill Agent: {agent_type}
Skill Prompt: {prompt_text}

Available agents and their capabilities:
{json.dumps(available, indent=2)}

Create a complete workflow plan that includes the skill execution as one of the
subtasks. Make sure the skill subtask has the exact description above.
Return a JSON object with a 'tasks' array where each task has
'description', 'agent_type', and 'depth' fields."""

        try:
            result = self.llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=system_prompt,
                temperature=0.3,
            )
        except Exception:
            result = self._fallback_skill_plan(skill_name, prompt_text, agent_type, available_agents)

        tasks_data = result.get("tasks", [])
        if not tasks_data:
            tasks_data = self._fallback_skill_plan(
                skill_name, prompt_text, agent_type, available_agents
            ).get("tasks", [])

        subtasks = []
        for i, task_data in enumerate(tasks_data):
            task_agent = task_data.get("agent_type", "").lower()
            if task_agent not in available_agents:
                task_agent = available_agents[0] if available_agents else agent_type

            subtask = Subtask(
                workflow_id=workflow_id,
                description=task_data.get(
                    "description", f"[SKILL: {skill_name}] {prompt_text[:50]}"
                ),
                agent_type=task_agent,
                status=SubtaskStatus.PENDING,
                depth=task_data.get("depth", 0),
                input_data=prompt_text,
                metadata={
                    "order": i,
                    "skill_name": skill_name,
                    "skill_agent": agent_type,
                    "from_skill": True,
                },
            )
            subtasks.append(subtask)

        return subtasks

    def _fallback_skill_plan(
        self, skill_name: str, prompt_text: str, agent_type: str, agents: List[str]
    ) -> Dict[str, Any]:
        """Generate a simple fallback plan for a skill prompt."""
        tasks = []
        agent_pretty = agent_type.title()

        if agent_type != "researcher" and "researcher" in agents:
            tasks.append({
                "description": f"Research background context for: {prompt_text[:80]}",
                "agent_type": "researcher",
                "depth": 0,
            })

        tasks.append({
            "description": f"[SKILL: {skill_name}] {prompt_text}",
            "agent_type": agent_type,
            "depth": 0,
        })

        if "reviewer" in agents and agent_type != "reviewer":
            tasks.append({
                "description": f"Review and improve the output of the {agent_pretty} skill",
                "agent_type": "reviewer",
                "depth": 1,
            })

        if "deliverer" in agents:
            tasks.append({
                "description": f"Deliver the final result from skill: {skill_name}",
                "agent_type": "deliverer",
                "depth": 1,
            })

        return {"tasks": tasks}

    def _fallback_plan(self, goal: str, agents: List[str]) -> Dict[str, Any]:
        """Generate a simple fallback plan when LLM is not available."""
        tasks = []

        if "researcher" in agents:
            tasks.append({
                "description": f"Research: {goal}",
                "agent_type": "researcher",
                "depth": 0,
            })

        # Handle custom agents — create a single subtask for each
        for a in agents:
            if a not in BUILTIN_AGENTS:
                tasks.append({
                    "description": f"Perform specialized task as {a}: {goal}",
                    "agent_type": a,
                    "depth": 0,
                })

        if "writer" in agents:
            tasks.append({
                "description": f"Write summary report for: {goal}",
                "agent_type": "writer",
                "depth": 0,
            })

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
