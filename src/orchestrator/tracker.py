"""Tracker — monitors progress of tasks and manages retries."""
from __future__ import annotations

from typing import Optional

from src.memory.models import SubtaskStatus, WorkflowStatus
from src.memory.task_store import TaskStore


class Tracker:
    """Tracks workflow and subtask progress, handles status updates."""

    def __init__(self, store: Optional[TaskStore] = None):
        self.store = store or TaskStore()

    def start_workflow(self, workflow_id: str) -> None:
        """Mark a workflow as running."""
        self.store.update_workflow_status(workflow_id, WorkflowStatus.RUNNING)

    def complete_workflow(self, workflow_id: str) -> None:
        """Mark a workflow as completed."""
        self.store.update_workflow_status(workflow_id, WorkflowStatus.COMPLETED)

    def fail_workflow(self, workflow_id: str) -> None:
        """Mark a workflow as failed."""
        self.store.update_workflow_status(workflow_id, WorkflowStatus.FAILED)

    def start_subtask(self, subtask_id: str) -> None:
        """Mark a subtask as running."""
        subtask = self.store.get_subtask(subtask_id)
        if subtask:
            subtask.update_status(SubtaskStatus.RUNNING)
            self.store.save_subtask(subtask)

    def complete_subtask(self, subtask_id: str, output: str) -> None:
        """Mark a subtask as completed with output."""
        subtask = self.store.get_subtask(subtask_id)
        if subtask:
            subtask.update_status(SubtaskStatus.COMPLETED)
            subtask.output_data = output
            self.store.save_subtask(subtask)

    def fail_subtask(self, subtask_id: str, error_message: str) -> None:
        """Mark a subtask as failed with error."""
        subtask = self.store.get_subtask(subtask_id)
        if subtask:
            subtask.update_status(SubtaskStatus.FAILED)
            subtask.error_message = error_message
            self.store.save_subtask(subtask)

    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowStatus]:
        """Get the current status of a workflow."""
        workflow = self.store.get_workflow(workflow_id)
        return workflow.status if workflow else None

    def get_progress(self, workflow_id: str) -> dict:
        """Get detailed progress information."""
        workflow = self.store.get_workflow(workflow_id)
        if not workflow:
            return {"error": "Workflow not found"}

        total = len(workflow.subtasks)
        completed = sum(1 for s in workflow.subtasks if s.status == SubtaskStatus.COMPLETED)
        failed = sum(1 for s in workflow.subtasks if s.status == SubtaskStatus.FAILED)
        running = sum(1 for s in workflow.subtasks if s.status == SubtaskStatus.RUNNING)
        pending = sum(1 for s in workflow.subtasks if s.status == SubtaskStatus.PENDING)

        return {
            "workflow_id": workflow_id,
            "goal": workflow.goal,
            "status": workflow.status.value,
            "total_subtasks": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "progress_pct": round((completed / total * 100) if total > 0 else 0, 1),
        }
