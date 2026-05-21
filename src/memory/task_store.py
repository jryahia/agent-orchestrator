"""SQLite-backed task store for persistence of workflows, subtasks, and task memory."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import Subtask, SubtaskStatus, Workflow, WorkflowStatus


class TaskStore:
    """Persistent store using SQLite for workflows and subtasks."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.environ.get(
                "SQLITE_PATH",
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "tasks.db")
            )
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                agents TEXT NOT NULL DEFAULT '[]',
                final_output TEXT,
                output_format TEXT NOT NULL DEFAULT 'markdown',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS subtasks (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                parent_id TEXT,
                description TEXT NOT NULL,
                agent_type TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                input_data TEXT,
                output_data TEXT,
                error_message TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                depth INTEGER NOT NULL DEFAULT 0,
                max_depth INTEGER NOT NULL DEFAULT 3,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (workflow_id) REFERENCES workflows(id)
            );

            CREATE INDEX IF NOT EXISTS idx_subtasks_workflow ON subtasks(workflow_id);
            CREATE INDEX IF NOT EXISTS idx_subtasks_status ON subtasks(status);
        """)
        conn.commit()

    def save_workflow(self, workflow: Workflow) -> None:
        conn = self._get_connection()
        conn.execute(
            """INSERT OR REPLACE INTO workflows
               (id, goal, status, agents, final_output, output_format, created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                workflow.id,
                workflow.goal,
                workflow.status.value,
                json.dumps(workflow.agents),
                workflow.final_output,
                workflow.output_format,
                workflow.created_at,
                workflow.updated_at,
                json.dumps(workflow.metadata),
            ),
        )
        conn.commit()

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,)).fetchone()
        if row is None:
            return None
        workflow = Workflow(
            id=row["id"],
            goal=row["goal"],
            status=WorkflowStatus(row["status"]),
            agents=json.loads(row["agents"]),
            final_output=row["final_output"],
            output_format=row["output_format"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]),
        )
        workflow.subtasks = self.get_subtasks(workflow_id)
        return workflow

    def list_workflows(self) -> List[Workflow]:
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM workflows ORDER BY created_at DESC").fetchall()
        workflows = []
        for row in rows:
            wf = Workflow(
                id=row["id"],
                goal=row["goal"],
                status=WorkflowStatus(row["status"]),
                agents=json.loads(row["agents"]),
                final_output=row["final_output"],
                output_format=row["output_format"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=json.loads(row["metadata"]),
            )
            wf.subtasks = self.get_subtasks(row["id"])
            workflows.append(wf)
        return workflows

    def save_subtask(self, subtask: Subtask) -> None:
        conn = self._get_connection()
        conn.execute(
            """INSERT OR REPLACE INTO subtasks
               (id, workflow_id, parent_id, description, agent_type, status,
                input_data, output_data, error_message,
                retry_count, max_retries, depth, max_depth,
                created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                subtask.id,
                subtask.workflow_id,
                subtask.parent_id,
                subtask.description,
                subtask.agent_type,
                subtask.status.value,
                subtask.input_data,
                subtask.output_data,
                subtask.error_message,
                subtask.retry_count,
                subtask.max_retries,
                subtask.depth,
                subtask.max_depth,
                subtask.created_at,
                subtask.updated_at,
                json.dumps(subtask.metadata),
            ),
        )
        conn.commit()

    def get_subtasks(self, workflow_id: str) -> List[Subtask]:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM subtasks WHERE workflow_id = ? ORDER BY created_at ASC",
            (workflow_id,),
        ).fetchall()
        return [
            Subtask(
                id=row["id"],
                workflow_id=row["workflow_id"],
                parent_id=row["parent_id"],
                description=row["description"],
                agent_type=row["agent_type"],
                status=SubtaskStatus(row["status"]),
                input_data=row["input_data"],
                output_data=row["output_data"],
                error_message=row["error_message"],
                retry_count=row["retry_count"],
                max_retries=row["max_retries"],
                depth=row["depth"],
                max_depth=row["max_depth"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]

    def get_subtask(self, subtask_id: str) -> Optional[Subtask]:
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM subtasks WHERE id = ?", (subtask_id,)).fetchone()
        if row is None:
            return None
        return Subtask(
            id=row["id"],
            workflow_id=row["workflow_id"],
            parent_id=row["parent_id"],
            description=row["description"],
            agent_type=row["agent_type"],
            status=SubtaskStatus(row["status"]),
            input_data=row["input_data"],
            output_data=row["output_data"],
            error_message=row["error_message"],
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            depth=row["depth"],
            max_depth=row["max_depth"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]),
        )

    def update_workflow_status(self, workflow_id: str, status: WorkflowStatus) -> None:
        conn = self._get_connection()
        conn.execute(
            "UPDATE workflows SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, datetime.now(timezone.utc).isoformat(), workflow_id),
        )
        conn.commit()

    def update_workflow_output(self, workflow_id: str, output: str) -> None:
        conn = self._get_connection()
        conn.execute(
            "UPDATE workflows SET final_output = ?, updated_at = ? WHERE id = ?",
            (output, datetime.now(timezone.utc).isoformat(), workflow_id),
        )
        conn.commit()

    def delete_workflow(self, workflow_id: str) -> None:
        conn = self._get_connection()
        conn.execute("DELETE FROM subtasks WHERE workflow_id = ?", (workflow_id,))
        conn.execute("DELETE FROM workflows WHERE id = ?", (workflow_id,))
        conn.commit()

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
