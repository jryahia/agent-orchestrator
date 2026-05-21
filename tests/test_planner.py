"""Tests for the Planner module."""
from __future__ import annotations

import os
import sys
import json
import tempfile
import unittest

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set dry-run mode
os.environ["DRY_RUN"] = "1"

from src.llm import LLMClient
from src.memory.models import SubtaskStatus
from src.orchestrator.planner import Planner


class TestPlanner(unittest.TestCase):
    """Test the Planner's ability to decompose goals into subtasks."""

    def setUp(self):
        self.llm = LLMClient()
        self.planner = Planner(self.llm)

    def test_plan_with_researcher_writer(self):
        """Test planning with researcher and writer agents."""
        goal = "Research Top 5 AI startups and write a summary report"
        agents = ["researcher", "writer"]

        subtasks = self.planner.plan(
            goal=goal,
            agents=agents,
            workflow_id="test-wf-1",
        )

        self.assertGreater(len(subtasks), 0, "Should produce at least one subtask")
        self.assertLessEqual(len(subtasks), 10, "Should not produce too many subtasks")

        # Check that agent types are valid
        for st in subtasks:
            self.assertIn(st.agent_type, agents, f"Agent type {st.agent_type} must be in {agents}")
            self.assertEqual(st.status, SubtaskStatus.PENDING)
            self.assertLessEqual(st.depth, st.max_depth)

        # Check ordering: researcher before writer
        agent_order = [st.agent_type for st in subtasks]
        researcher_positions = [i for i, a in enumerate(agent_order) if a == "researcher"]
        writer_positions = [i for i, a in enumerate(agent_order) if a == "writer"]

        if researcher_positions and writer_positions:
            self.assertLess(
                max(researcher_positions),
                min(writer_positions),
                "Researcher should come before writer",
            )

    def test_plan_with_all_agents(self):
        """Test planning with all agent types."""
        goal = "Create a comprehensive report on quantum computing advancements"
        agents = ["researcher", "writer", "reviewer", "deliverer"]

        subtasks = self.planner.plan(
            goal=goal,
            agents=agents,
            workflow_id="test-wf-2",
        )

        self.assertGreater(len(subtasks), 0)
        agent_types_used = set(st.agent_type for st in subtasks)
        # All agent types used should be valid (subset of available)
        for a in agent_types_used:
            self.assertIn(a, agents, f"Agent type {a} must be in available {agents}")
        # At minimum, researcher and writer should be planned
        self.assertIn("researcher", agent_types_used)
        self.assertIn("writer", agent_types_used)

    def test_plan_with_single_agent(self):
        """Test planning with only one agent type."""
        goal = "Find information about Python 3.14"
        agents = ["researcher"]

        subtasks = self.planner.plan(
            goal=goal,
            agents=agents,
            workflow_id="test-wf-3",
        )

        self.assertGreater(len(subtasks), 0)
        for st in subtasks:
            self.assertEqual(st.agent_type, "researcher")

    def test_plan_max_depth_respected(self):
        """Test that max depth is respected."""
        goal = "Write a deep research paper"
        agents = ["researcher", "writer"]

        subtasks = self.planner.plan(
            goal=goal,
            agents=agents,
            workflow_id="test-wf-4",
            max_depth=1,
        )

        for st in subtasks:
            self.assertLessEqual(st.depth, 1)

    def test_fallback_plan(self):
        """Test the fallback plan when LLM is not available."""
        goal = "Test fallback planning"
        agents = ["researcher", "writer", "reviewer"]

        result = self.planner._fallback_plan(goal, agents)
        tasks = result.get("tasks", [])

        self.assertEqual(len(tasks), 3)
        self.assertEqual(tasks[0]["agent_type"], "researcher")
        self.assertEqual(tasks[1]["agent_type"], "writer")
        self.assertEqual(tasks[2]["agent_type"], "reviewer")


if __name__ == "__main__":
    unittest.main()
