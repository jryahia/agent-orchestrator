#!/usr/bin/env python3
"""
Agent Orchestrator — Multi-Agent Workflow System
CLI entry point for running multi-agent workflows.

Usage:
    python main.py "Research Top 5 AI startups and write a summary report"
    python main.py --file tasks.json --parallel
    python main.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

# Ensure the project root is on sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.llm import LLMClient
from src.memory.models import Workflow
from src.memory.task_store import TaskStore
from src.orchestrator import Dispatcher, Planner, Tracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent Orchestrator — Multi-Agent Workflow System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py "Research Top 5 AI startups and write a summary report"
              python main.py --file tasks.json --parallel
              python main.py --goal "Summarize Python 3.14 features" --agents researcher writer --output markdown
        """),
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("goal", nargs="?", type=str, help="The goal for the workflow")
    input_group.add_argument("--goal", type=str, help="The goal for the workflow (alternative)")
    input_group.add_argument("-f", "--file", type=str, help="JSON file with goals list")

    parser.add_argument("--agents", nargs="+", default=["researcher", "writer", "reviewer"],
                        choices=["researcher", "writer", "reviewer", "deliverer"],
                        help="Agents to use in the workflow")
    parser.add_argument("--output", type=str, choices=["markdown", "json", "text"],
                        default="markdown", help="Output format")
    parser.add_argument("--parallel", action="store_true", help="Execute tasks in parallel where possible")
    parser.add_argument("--dry-run", action="store_true", help="Test without real LLM calls or web searches")
    parser.add_argument("--db", type=str, help="Path to SQLite database file")
    parser.add_argument("--api", action="store_true", help="Start the API server instead of CLI mode")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")
    parser.add_argument("--output-file", type=str, help="Save output to a file")

    return parser.parse_args()


def run_single_workflow(
    goal: str,
    agents: list,
    output_format: str,
    parallel: bool,
    dry_run: bool,
    output_file: str = None,
) -> dict:
    """Execute a single workflow and return results."""
    if dry_run:
        os.environ["DRY_RUN"] = "1"
        print(f"🔧 DRY RUN MODE — no real LLM calls or web searches will be made\n")

    store = TaskStore()
    llm = LLMClient()
    planner = Planner(llm)
    tracker = Tracker(store)
    dispatcher = Dispatcher(task_store=store, llm_client=llm, tracker=tracker, parallel=parallel)

    workflow = Workflow(goal=goal, agents=agents, output_format=output_format)
    store.save_workflow(workflow)

    print(f"📋 Workflow ID: {workflow.id}")
    print(f"🎯 Goal: {goal}")
    print(f"🤖 Agents: {', '.join(agents)}")
    print(f"📊 Output format: {output_format}")
    print(f"{'⚡ Parallel execution' if parallel else '➡️ Sequential execution'}")
    print()

    # Plan
    print("📝 Planning subtasks...")
    subtasks = planner.plan(
        goal=goal,
        agents=agents,
        workflow_id=workflow.id,
    )

    for st in subtasks:
        workflow.add_subtask(st)
        store.save_subtask(st)

    print(f"  Created {len(subtasks)} subtask(s):")
    for st in subtasks:
        print(f"    • [{st.agent_type}] {st.description[:80]}...")
    print()

    # Execute
    print("🚀 Executing workflow...")
    result = dispatcher.execute_workflow(workflow.id, subtasks)

    print()
    print("=" * 60)
    print("✅ WORKFLOW COMPLETE")
    print("=" * 60)
    print()

    final_output = result.get("final_output", "")

    # Display output
    if output_format == "json":
        print(json.dumps({"goal": goal, "result": final_output}, indent=2))
    else:
        print(final_output)

    # Save to file if requested
    if output_file:
        ext = {"markdown": ".md", "json": ".json", "text": ".txt"}.get(output_format, ".md")
        fname = output_file if output_file.endswith(ext) else output_file + ext
        with open(fname, "w", encoding="utf-8") as f:
            f.write(final_output)
        print(f"\n💾 Output saved to: {fname}")

    # Show progress summary
    progress = tracker.get_progress(workflow.id)
    print(f"\n📊 Progress: {progress['completed']}/{progress['total_subtasks']} subtasks completed")
    if progress["failed"] > 0:
        print(f"❌ Failed: {progress['failed']} subtask(s)")

    return result


def run_file_workflow(file_path: str, agents: list, parallel: bool, dry_run: bool):
    """Run workflows from a JSON file containing multiple goals."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    goals = data if isinstance(data, list) else data.get("goals", [data])
    print(f"📋 Loaded {len(goals)} goal(s) from {file_path}\n")

    for i, entry in enumerate(goals):
        goal = entry if isinstance(entry, str) else entry.get("goal", str(entry))
        goal_agents = agents or entry.get("agents", ["researcher", "writer"])
        goal_output = entry.get("output_format", "markdown")

        print(f"{'='*60}")
        print(f"Workflow {i+1}/{len(goals)}")
        print(f"{'='*60}")
        run_single_workflow(goal, goal_agents, goal_output, parallel, dry_run)
        print()


def start_api_server(port: int):
    """Start the FastAPI API server."""
    print(f"🌐 Starting API server on port {port}...")
    print(f"📖 API docs at http://localhost:{port}/docs")
    print(f"💚 Health check at http://localhost:{port}/health")
    print()
    from api import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)


def main():
    args = parse_args()

    # Resolve goal from positional or --goal
    goal = args.goal or args.goal_positional

    if args.api:
        start_api_server(args.port)
        return

    if args.file:
        run_file_workflow(args.file, args.agents, args.parallel, args.dry_run)
        return

    if not goal:
        print("❌ Error: No goal provided. Use a positional argument or --goal.")
        print("   Example: python main.py \"Research AI trends\"")
        sys.exit(1)

    run_single_workflow(goal, args.agents, args.output, args.parallel, args.dry_run, args.output_file)


if __name__ == "__main__":
    main()
