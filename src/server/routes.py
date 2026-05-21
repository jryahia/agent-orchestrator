"""API server routes for the agent orchestrator."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException

from src.memory.models import WorkflowCreate, WorkflowResponse
from src.memory.task_store import TaskStore
from src.orchestrator import Dispatcher, Planner, Tracker

router = APIRouter(prefix="/api/v1", tags=["workflows"])


def get_store() -> TaskStore:
    """Get or create a task store instance."""
    if not hasattr(get_store, "_store"):
        get_store._store = TaskStore()
    return get_store._store


@router.post("/workflows", response_model=WorkflowResponse, status_code=201)
async def create_workflow(body: WorkflowCreate):
    """Create and execute a new workflow."""
    store = get_store()
    planner = Planner()
    tracker = Tracker(store)
    dispatcher = Dispatcher(task_store=store, tracker=tracker)

    from src.memory.models import Workflow

    workflow = Workflow(
        goal=body.goal,
        agents=body.agents,
        output_format=body.output_format,
    )

    # Plan subtasks
    subtasks = planner.plan(
        goal=body.goal,
        agents=body.agents,
        workflow_id=workflow.id,
    )

    for st in subtasks:
        workflow.add_subtask(st)

    store.save_workflow(workflow)

    # Execute (in background for real cases; here inline for simplicity)
    try:
        result = dispatcher.execute_workflow(workflow.id, subtasks)
        updated = store.get_workflow(workflow.id)
        if updated:
            return WorkflowResponse(
                id=updated.id,
                goal=updated.goal,
                status=updated.status,
                subtasks=updated.subtasks,
                final_output=updated.final_output,
                created_at=updated.created_at,
                updated_at=updated.updated_at,
            )
    except Exception as e:
        tracker.fail_workflow(workflow.id)
        raise HTTPException(status_code=500, detail=str(e))

    return WorkflowResponse(
        id=workflow.id,
        goal=workflow.goal,
        status=workflow.status,
        subtasks=workflow.subtasks,
        final_output=workflow.final_output,
        created_at=workflow.created_at,
        updated_at=workflow.updated_at,
    )


@router.get("/workflows", response_model=List[WorkflowResponse])
async def list_workflows():
    """List all workflows."""
    store = get_store()
    workflows = store.list_workflows()
    return [
        WorkflowResponse(
            id=wf.id,
            goal=wf.goal,
            status=wf.status,
            subtasks=wf.subtasks,
            final_output=wf.final_output,
            created_at=wf.created_at,
            updated_at=wf.updated_at,
        )
        for wf in workflows
    ]


@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str):
    """Get workflow details by ID."""
    store = get_store()
    workflow = store.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return WorkflowResponse(
        id=workflow.id,
        goal=workflow.goal,
        status=workflow.status,
        subtasks=workflow.subtasks,
        final_output=workflow.final_output,
        created_at=workflow.created_at,
        updated_at=workflow.updated_at,
    )


@router.get("/workflows/{workflow_id}/result")
async def get_workflow_result(workflow_id: str, format: Optional[str] = None):
    """Get the final result of a workflow."""
    store = get_store()
    workflow = store.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.status.value != "completed":
        raise HTTPException(status_code=400, detail="Workflow has not completed yet")

    output_format = format or workflow.output_format
    content = workflow.final_output or ""

    if output_format == "json":
        from fastapi.responses import JSONResponse
        return JSONResponse({"id": workflow_id, "goal": workflow.goal, "result": content})
    elif output_format == "text":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content)
    else:
        from fastapi.responses import HTMLResponse
        import markdown as md_lib
        html = md_lib.markdown(content)
        return HTMLResponse(f"<html><body>{html}</body></html>")


@router.get("/workflows/{workflow_id}/progress")
async def get_workflow_progress(workflow_id: str):
    """Get detailed progress of a workflow."""
    store = get_store()
    tracker = Tracker(store)
    progress = tracker.get_progress(workflow_id)
    if "error" in progress:
        raise HTTPException(status_code=404, detail=progress["error"])
    return progress


@router.delete("/workflows/{workflow_id}", status_code=204)
async def delete_workflow(workflow_id: str):
    """Delete a workflow."""
    store = get_store()
    workflow = store.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    store.delete_workflow(workflow_id)
    return None
