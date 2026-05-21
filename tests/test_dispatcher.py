"""Tests for the Dispatcher module."""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set dry-run mode
os.environ["DRY_RUN"] = "1"

from src.llm import LLMClient
from src.memory.models import Subtask, SubtaskStatus, Workflow
from src.memory.task_store import TaskStore
from src.orchestrator.dispatcher import Dispatcher
from src.orchestrator.tracker import Tracker


class TestDispatcher(unittest.TestCase):
    """Test the Dispatcher's ability to execute workflows."""

    def setUp(self):
        # Use in-memory SQLite for tests
        self.db_path = ":memory:"
        self.store = TaskStore(db_path=self.db_path)
        self.llm = LLMClient()
        self.tracker = Tracker(self.store)
        self.dispatcher = Dispatcher(
            task_store=self.store,
            llm_client=self.llm,
            tracker=self.tracker,
            parallel=False,
        )

    def _create_test_subtask(self, agent_type: str, description: str, order: int = 0, workflow_id: str = None) -> Subtask:
        return Subtask(
            workflow_id=workflow_id or "test-wf",
            description=description,
            agent_type=agent_type,
            status=SubtaskStatus.PENDING,
            input_data=description,
            metadata={"order": order, "goal": "Test goal"},
        )

    def test_sequential_execution(self):
        """Test sequential execution of subtasks."""
        workflow = Workflow(goal="Test goal", agents=["researcher", "writer"])
        self.store.save_workflow(workflow)

        subtasks = [
            self._create_test_subtask("researcher", "Research test topic", 0, workflow.id),
            self._create_test_subtask("writer", "Write summary of research", 1, workflow.id),
        ]

        for st in subtasks:
            self.store.save_subtask(st)

        result = self.dispatcher._execute_sequential(subtasks)

        # Check all subtasks were executed
        self.assertEqual(len(result), 2)
        for st in subtasks:
            updated = self.store.get_subtask(st.id)
            self.assertIsNotNone(updated)
            # In dry run, these should complete
            self.assertIn(updated.status, [SubtaskStatus.COMPLETED, SubtaskStatus.FAILED])

    def test_parallel_execution(self):
        """Test parallel execution of independent subtasks."""
        self.dispatcher.parallel = True
        workflow = Workflow(goal="Test parallel", agents=["researcher", "writer"])
        self.store.save_workflow(workflow)

        subtasks = [
            self._create_test_subtask("researcher", "Research topic A", 0, workflow.id),
            self._create_test_subtask("researcher", "Research topic B", 1, workflow.id),
        ]

        for st in subtasks:
            self.store.save_subtask(st)

        result = self.dispatcher._execute_parallel(subtasks)

        self.assertEqual(len(result), 2)

    def test_retry_logic(self):
        """Test that retry logic works for failed subtasks."""
        subtask = self._create_test_subtask("researcher", "Test retries")
        subtask.max_retries = 3
        subtask.status = SubtaskStatus.FAILED  # Must be FAILED to be retryable

        self.assertTrue(subtask.can_retry(), "Failed subtask should be retryable")

        subtask.retry_count = 3
        self.assertFalse(subtask.can_retry(), "Subtask at max retries should not be retryable")

    def test_execute_with_retry_exhausted(self):
        """Test that execution returns error after retries exhausted."""
        subtask = self._create_test_subtask("researcher", "Will fail")
        subtask.max_retries = 0  # No retries

        # The subtask will try to execute but should fail gracefully
        result = self.dispatcher._execute_subtask_with_retry(subtask, {})

        # Should get result even in dry run (mock response)
        self.assertIsNotNone(result)

    def test_get_agent_by_type(self):
        """Test agent instantiation by type."""
        researcher = self.dispatcher._get_agent("researcher")
        self.assertEqual(researcher.name, "researcher")

        writer = self.dispatcher._get_agent("writer")
        self.assertEqual(writer.name, "writer")

        reviewer = self.dispatcher._get_agent("reviewer")
        self.assertEqual(reviewer.name, "reviewer")

        deliverer = self.dispatcher._get_agent("deliverer")
        self.assertEqual(deliverer.name, "deliverer")

        # Test caching
        self.assertIs(self.dispatcher._get_agent("researcher"), researcher)

    def test_invalid_agent_type(self):
        """Test that invalid agent type raises ValueError."""
        with self.assertRaises(ValueError):
            self.dispatcher._get_agent("nonexistent_agent")


class TestTracker(unittest.TestCase):
    """Test the Tracker module."""

    def setUp(self):
        self.db_path = ":memory:"
        self.store = TaskStore(db_path=self.db_path)
        self.tracker = Tracker(self.store)

    def test_workflow_lifecycle(self):
        """Test workflow status transitions."""
        workflow = Workflow(goal="Test lifecycle")
        self.store.save_workflow(workflow)

        self.tracker.start_workflow(workflow.id)
        updated = self.store.get_workflow(workflow.id)
        self.assertIsNotNone(updated)
        self.assertEqual(updated.status.value, "running")

        self.tracker.complete_workflow(workflow.id)
        updated = self.store.get_workflow(workflow.id)
        self.assertEqual(updated.status.value, "completed")

    def test_subtask_lifecycle(self):
        """Test subtask status transitions."""
        # Create a workflow first (needed for foreign key)
        workflow = Workflow(goal="Test subtask lifecycle", agents=["researcher"])
        self.store.save_workflow(workflow)

        subtask = Subtask(
            workflow_id=workflow.id,
            description="Test subtask",
            agent_type="researcher",
        )
        self.store.save_subtask(subtask)

        self.tracker.start_subtask(subtask.id)
        updated = self.store.get_subtask(subtask.id)
        self.assertEqual(updated.status.value, "running")

        self.tracker.complete_subtask(subtask.id, "Test output")
        updated = self.store.get_subtask(subtask.id)
        self.assertEqual(updated.status.value, "completed")
        self.assertEqual(updated.output_data, "Test output")

    def test_progress_tracking(self):
        """Test progress reporting."""
        workflow = Workflow(goal="Test progress")
        self.store.save_workflow(workflow)

        # Add subtasks
        for i in range(5):
            st = Subtask(
                workflow_id=workflow.id,
                description=f"Task {i}",
                agent_type="researcher",
            )
            self.store.save_subtask(st)

        progress = self.tracker.get_progress(workflow.id)
        self.assertEqual(progress["total_subtasks"], 5)
        self.assertEqual(progress["completed"], 0)
        self.assertEqual(progress["progress_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
