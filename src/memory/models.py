"""Pydantic schemas for the agent orchestrator system."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SubtaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Subtask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    parent_id: Optional[str] = None
    description: str
    agent_type: str = ""  # researcher, writer, reviewer, deliverer
    status: SubtaskStatus = SubtaskStatus.PENDING
    input_data: Optional[str] = None
    output_data: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    depth: int = 0
    max_depth: int = 3
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def update_status(self, status: SubtaskStatus) -> None:
        self.status = status
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def can_retry(self) -> bool:
        return self.status == SubtaskStatus.FAILED and self.retry_count < self.max_retries

    def increment_retry(self) -> None:
        self.retry_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()


class Workflow(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    subtasks: List[Subtask] = Field(default_factory=list)
    agents: List[str] = Field(default_factory=lambda: ["researcher", "writer", "reviewer"])
    final_output: Optional[str] = None
    output_format: str = "markdown"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def update_status(self, status: WorkflowStatus) -> None:
        self.status = status
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_subtask(self, subtask: Subtask) -> None:
        subtask.workflow_id = self.id
        self.subtasks.append(subtask)
        self.updated_at = datetime.now(timezone.utc).isoformat()


class Task(BaseModel):
    """A plain task model for simple task execution (non-workflow)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    status: SubtaskStatus = SubtaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WorkflowCreate(BaseModel):
    goal: str
    agents: List[str] = Field(default_factory=lambda: ["researcher", "writer"])
    output_format: str = "markdown"


class WorkflowResponse(BaseModel):
    id: str
    goal: str
    status: WorkflowStatus
    subtasks: List[Subtask]
    final_output: Optional[str] = None
    created_at: str
    updated_at: str
