#!/usr/bin/env python3
"""
Interactive TUI for Agent Orchestrator — Rich-powered terminal interface.

Usage:
    .venv/Scripts/python interactive.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Ensure project root on path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from src.llm import LLMClient
from src.memory.models import (
    Subtask,
    SubtaskStatus,
    Workflow,
    WorkflowCreate,
    WorkflowStatus,
)
from src.memory.task_store import TaskStore
from src.orchestrator import Dispatcher, Planner, Tracker
from src.custom_agent_store import CustomAgentStore
from src.skills_store import SkillsStore

# ── Console ──────────────────────────────────────────────────────────────────

console = Console()

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_FILE = os.path.join(project_root, ".orchestrator_config.json")
SKILLS_FILE = os.path.join(project_root, ".orchestrator_skills.json")

DEFAULT_CONFIG = {
    "agents": ["researcher", "writer", "reviewer"],
    "output_format": "markdown",
    "parallel": False,
    "db_path": os.path.join(project_root, "data", "tasks.db"),
    "api_key": "",
    "base_url": "",
    "model": "deepseek-chat",
}


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


# ── Header ────────────────────────────────────────────────────────────────────

def make_header(title: str, subtitle: str = "") -> Panel:
    t = Text()
    t.append("  ╔══════════════════════════════════════╗\n", style="bright_black")
    t.append(f"  ║  {title:<35}║\n", style="bold cyan")
    t.append("  ╚══════════════════════════════════════╝", style="bright_black")
    if subtitle:
        t.append(f"\n  {subtitle}", style="italic dim white")
    return Panel(t, box=box.MINIMAL, border_style="cyan", padding=(0, 1))


# ── Agent labels ──────────────────────────────────────────────────────────────

AGENT_MARK = {
    "researcher": "•",
    "writer": "•",
    "reviewer": "•",
    "deliverer": "•",
}
AGENT_COLORS = {
    "researcher": "blue",
    "writer": "green",
    "reviewer": "yellow",
    "deliverer": "magenta",
}


def agent_tag(agent_type: str) -> Text:
    mark = AGENT_MARK.get(agent_type, "•")
    color = AGENT_COLORS.get(agent_type, "bright_cyan")
    return Text(f"{mark} {agent_type.title()}", style=f"bold {color}")


# ── History ───────────────────────────────────────────────────────────────────

def view_history(store: TaskStore) -> None:
    """Show past workflows in a table."""
    workflows = store.list_workflows()
    if not workflows:
        console.print("\n[dim]No past workflows found.[/dim]\n")
        return

    table = Table(box=box.ROUNDED, border_style="cyan", header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Goal", width=40, no_wrap=True)
    table.add_column("Status", width=12)
    table.add_column("Subtask Split", width=14)
    table.add_column("When", width=20)

    for i, wf in enumerate(workflows[:20], 1):
        status_style = {
            "completed": "green",
            "failed": "red",
            "running": "yellow",
            "pending": "dim",
        }.get(wf.status.value, "white")
        status_label = {
            "completed": "Completed",
            "failed": "Failed",
            "running": "Running",
            "pending": "Pending",
        }.get(wf.status.value, wf.status.value)

        done = sum(1 for s in wf.subtasks if s.status == SubtaskStatus.COMPLETED)
        total = len(wf.subtasks)
        subtask_str = f"{done}/{total}" if total else "—"

        created = wf.created_at[:19] if wf.created_at else "—"

        table.add_row(
            str(i),
            wf.goal[:38] + "…" if len(wf.goal) > 38 else wf.goal,
            Text(status_label, style=status_style),
            Text(subtask_str, style="bold" if done == total and total else ""),
            created,
        )

    console.print("\n")
    console.print(table)

    # Option to view details
    choice = Prompt.ask(
        "\n[dim]Enter # to view result, or leave blank to go back[/dim]",
        default="",
    )
    if choice and choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(workflows):
            _show_workflow_detail(workflows[idx])


def _show_workflow_detail(wf: Workflow) -> None:
    """Show the full result of a past workflow."""
    console.clear()
    console.print(make_header("Workflow Result", wf.goal[:60]))
    console.print()

    table = Table(box=box.SIMPLE, border_style="dim")
    table.add_column("Agent", width=16)
    table.add_column("Status", width=12)
    table.add_column("Description", width=50)
    for st in wf.subtasks:
        s = "✓" if st.status == SubtaskStatus.COMPLETED else "✗" if st.status == SubtaskStatus.FAILED else "·"
        table.add_row(str(agent_tag(st.agent_type)), f"{s} {st.status.value}", st.description[:47] + "…" if len(st.description) > 47 else st.description)
    console.print(table)

    if wf.final_output:
        console.print("\n[bold cyan]Final Output:[/bold cyan]\n")
        preview = wf.final_output[:2000]
        syntax = Syntax(preview, "markdown", theme="dracula", word_wrap=True)
        console.print(Panel(syntax, border_style="dim", title="Result"))
        if len(wf.final_output) > 2000:
            console.print("[dim]… (truncated)[/dim]")

    Prompt.ask("\n[dim]Press Enter to go back[/dim]")


# ── Settings ──────────────────────────────────────────────────────────────────

def edit_settings(cfg: dict) -> dict:
    """Edit configuration interactively."""
    console.clear()
    console.print(make_header("Settings"))

    console.print("\n[bold]Agents to use:[/bold]")
    # Include custom agents alongside built-in ones
    custom_store = CustomAgentStore(data_dir=project_root)
    custom_agents = [a.name for a in custom_store.list()]
    all_agents = ["researcher", "writer", "reviewer", "deliverer"] + custom_agents
    current = cfg.get("agents", ["researcher", "writer", "reviewer"])
    for a in all_agents:
        checked = "(x)" if a in current else "( )"
        console.print(f"  {checked} {AGENT_MARK.get(a, '•')} {a.title()}")

    suggested_default = ", ".join(current)
    console.print("\n[dim]Available: researcher, writer, reviewer, deliverer" + (f", {', '.join(custom_agents)}" if custom_agents else "") + "[/dim]")
    choice = Prompt.ask(
        "\n[dim]Enter agents separated by commas[/dim]",
        default=suggested_default,
    )
    cfg["agents"] = [a.strip() for a in choice.split(",") if a.strip() in all_agents]
    if not cfg["agents"]:
        cfg["agents"] = ["researcher", "writer"]

    fmt = Prompt.ask(
        "[dim]Output format[/dim]",
        choices=["markdown", "json", "text"],
        default=cfg.get("output_format", "markdown"),
    )
    cfg["output_format"] = fmt

    cfg["parallel"] = Confirm.ask("[dim]Run tasks in parallel?[/dim]", default=cfg.get("parallel", False))

    # ── LLM Configuration ──────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]LLM Configuration[/bold cyan]", style="cyan"))

    # Provider selection with auto-fill of base_url
    PROVIDERS = {
        "OpenAI": "https://api.openai.com/v1",
        "DeepSeek": "https://api.deepseek.com",
        "OpenRouter": "https://openrouter.ai/api/v1",
        "Custom": None,
    }

    # Determine current provider from saved base_url
    current_base = cfg.get("base_url", "")
    provider_choices = list(PROVIDERS.keys())
    default_provider = "Custom"
    for pname, purl in PROVIDERS.items():
        if purl and current_base.rstrip("/") == purl.rstrip("/"):
            default_provider = pname
            break

    provider = Prompt.ask(
        "[bold]API Provider[/bold]",
        choices=provider_choices,
        default=default_provider,
    )

    if provider == "Custom":
        cfg["base_url"] = Prompt.ask(
            "[bold]Base URL[/bold]",
            default=current_base or "https://api.openai.com/v1",
        )
    else:
        cfg["base_url"] = PROVIDERS[provider]

    # Model selection
    cfg["model"] = Prompt.ask(
        "[bold]Model[/bold]",
        default=cfg.get("model", "deepseek-chat"),
    )

    # API Key input — always ask, show masked current status
    current_key = cfg.get("api_key", "") or ""
    if current_key:
        masked = current_key[:3] + "****" + current_key[-4:] if len(current_key) > 7 else "****"
        console.print(f"  [dim]Current API key: {masked}[/dim]")
    else:
        console.print("  [dim]No API key set (will use LLM_API_KEY env var)[/dim]")

    new_key = Prompt.ask(
        "[bold]API Key[/bold] (leave blank to keep current)",
        default="",
    )
    if new_key.strip():
        cfg["api_key"] = new_key.strip()

    save_config(cfg)
    console.print("\n[green]✓ Settings saved[/green]")
    time.sleep(0.8)
    return cfg


# ── Workflow progress ─────────────────────────────────────────────────────────

class ProgressTracker:
    """Hooks into the dispatcher to show live progress."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.running = 0
        self._lock = threading.Lock()
        self._events: List[str] = []

    def on_task_start(self, subtask_id: str, agent: str, desc: str) -> None:
        with self._lock:
            self.running += 1
            self._events.append(f"▶️  {agent_tag(agent)} started: {desc[:50]}")
            self._render()

    def on_task_done(self, subtask_id: str, agent: str, desc: str, status: SubtaskStatus) -> None:
        with self._lock:
            self.running -= 1
            if status == SubtaskStatus.COMPLETED:
                self.completed += 1
                icon = "✓"
            else:
                self.failed += 1
                icon = "✗"
            self._events.append(f"{icon} {agent_tag(agent)} {status.value}: {desc[:50]}")
            self._render()

    def _render(self) -> None:
        pass  # We'll use a callback-based approach

    @property
    def progress_pct(self) -> float:
        done = self.completed + self.failed
        return (done / self.total * 100) if self.total else 0


# ── Run workflow interactive ──────────────────────────────────────────────────

def run_workflow_interactive(goal: str, cfg: dict) -> None:
    """Run a workflow with live progress display."""
    from rich.markdown import Markdown

    console.clear()
    console.print(make_header("Running Workflow", goal[:70]))
    console.print()

    store = TaskStore(db_path=cfg.get("db_path"))
    llm = LLMClient(
        api_key=cfg.get("api_key"),
        base_url=cfg.get("base_url"),
        model=cfg.get("model"),
    )
    planner = Planner(llm)
    tracker = Tracker(store)
    dispatcher = Dispatcher(
        task_store=store, llm_client=llm, tracker=tracker, parallel=cfg.get("parallel", False),
        custom_agents_dir=project_root,
    )

    # Plan
    with console.status("[bold cyan]Planning subtasks…[/bold cyan]", spinner="dots"):
        workflow = Workflow(
            goal=goal,
            agents=cfg.get("agents", ["researcher", "writer", "reviewer"]),
            output_format=cfg.get("output_format", "markdown"),
        )
        store.save_workflow(workflow)
        subtasks = planner.plan(
            goal=goal,
            agents=workflow.agents,
            workflow_id=workflow.id,
        )
        for st in subtasks:
            workflow.add_subtask(st)
            store.save_subtask(st)

    # Show plan
    plan_table = Table(box=box.SIMPLE, border_style="dim", padding=(0, 1))
    plan_table.add_column("Step", style="bold dim", width=5)
    plan_table.add_column("Agent", width=18)
    plan_table.add_column("Task", width=70)
    for i, st in enumerate(subtasks, 1):
        plan_table.add_row(str(i), str(agent_tag(st.agent_type)), st.description)
    console.print(Panel(plan_table, title="Plan", border_style="cyan"))
    console.print()

    # Progress bar
    total_steps = len(subtasks)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    task_id = progress.add_task("[cyan]Executing…", total=total_steps)

    event_log: List[str] = []

    with progress:
        for i, st in enumerate(subtasks):
            agent_label = str(agent_tag(st.agent_type))
            desc_short = st.description[:60]

            progress.update(task_id, description=f"[cyan]{agent_label}: {desc_short}")

            # Execute
            try:
                result = dispatcher._execute_subtask_with_retry(st, {"previous_results": ""})
                store.save_subtask(st)
                status = store.get_subtask(st.id)
                status_val = status.status if status else SubtaskStatus.COMPLETED
                icon = "✓" if status_val == SubtaskStatus.COMPLETED else "✗"
            except Exception as e:
                result = f"Error: {e}"
                icon = "✗"

            event_log.append(f"  {icon} {agent_label}: {desc_short}")
            progress.update(task_id, advance=1)

        progress.update(task_id, description="[bold green]Workflow Complete![/bold green]")

    # Compile
    final_output = dispatcher._compile_results(subtasks, {st.id: st.output_data or "" for st in subtasks})
    store.update_workflow_output(workflow.id, final_output)
    tracker.complete_workflow(workflow.id)

    console.print()

    # Result
    console.print(Rule(style="cyan"))
    console.print("[bold cyan]Result[/bold cyan]")
    console.print()

    if cfg.get("output_format") == "json":
        try:
            data = json.loads(final_output)
            console.print_json(data=data)
        except Exception:
            console.print(final_output[:3000])
    else:
        md = Markdown(final_output[:3000])
        console.print(Panel(md, border_style="dim", title="Output", padding=(1, 2)))

    if len(final_output) > 3000:
        console.print("\n[dim]… (output truncated in preview)[/dim]")

    # Save option
    if Confirm.ask("\n[dim]Save output to file?[/dim]", default=False):
        fname = f"output_{workflow.id[:8]}.md"
        with open(os.path.join(project_root, fname), "w") as f:
            f.write(final_output)
        console.print(f"[green]✓ Saved to {fname}[/green]")

    console.print()
    console.print(Panel(
        f"[bold]Workflow ID:[/bold] {workflow.id}\n"
        f"[bold]Status:[/bold] {total_steps}/{total_steps} subtasks completed\n"
        f"[bold]Goal:[/bold] {goal}",
        border_style="green",
        title="Done",
    ))
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


# ── Quick goal input ──────────────────────────────────────────────────────────

def show_goal_prompt(cfg: dict) -> None:
    """Interactive goal input with suggestions."""
    console.clear()
    console.print(make_header("New Workflow", "Type your goal or pick a template"))

    suggestions = [
        "Research the top 5 open-source LLMs in 2025 and write a comparison report",
        "Find the latest Python web framework trends and summarize them",
        "Compare Cursor, Windsurf, and GitHub Copilot with a recommendation",
        "Research AI coding assistants and write a buying guide",
        "Custom goal…",
        "Use a saved skill…",
    ]

    for i, s in enumerate(suggestions, 1):
        console.print(f"  [dim]{i}.[/dim] {s}")

    choice = Prompt.ask("\n[bold cyan]Your choice[/bold cyan]", default="5")

    if choice == "6":
        # Forward to skills menu's "Use a Skill"
        store = _get_skills_store()
        _use_skill(store, cfg)
        return
    if choice == "5" or choice not in [str(i) for i in range(1, 7)]:
        goal = Prompt.ask("[bold cyan]Your goal[/bold cyan]")
    else:
        goal = suggestions[int(choice) - 1]

    if not goal.strip():
        console.print("[red]Goal cannot be empty.[/red]")
        return

    run_workflow_interactive(goal.strip(), cfg)


# ── Skills management ────────────────────────────────────────────────────────

def _get_skills_store() -> SkillsStore:
    """Get the SkillsStore instance."""
    return SkillsStore(SKILLS_FILE)


def show_skills_menu(cfg: dict) -> None:
    """Display the skills management sub-menu."""
    store = _get_skills_store()

    while True:
        console.clear()
        console.print(make_header("Skills", "Manage reusable prompt skills"))
        console.print()

        menu = Table(box=box.ROUNDED, border_style="cyan", show_header=False, padding=(0, 2))
        menu.add_column("Option", style="bold", width=4)
        menu.add_column("Action", width=50)
        menu.add_row("1", "List Skills  —  Show all saved prompt skills")
        menu.add_row("2", "Create Skill  —  Save a new reusable prompt")
        menu.add_row("3", "Delete Skill  —  Remove a saved skill")
        menu.add_row("4", "Use Skill  —  Run a workflow with a saved skill")
        menu.add_row("5", "Back to Main Menu")
        console.print(menu)
        console.print()

        choice = Prompt.ask(
            "[bold cyan]Select option[/bold cyan]",
            choices=["1", "2", "3", "4", "5"],
            default="5",
        )

        if choice == "1":
            _list_skills(store)
        elif choice == "2":
            _create_skill(store)
        elif choice == "3":
            _delete_skill(store)
        elif choice == "4":
            _use_skill(store, cfg)
        elif choice == "5":
            break


def _show_skills_table(store: SkillsStore) -> bool:
    """Render a table of saved skills. Returns True if any skills exist."""
    skills = store.list_skills()
    if not skills:
        console.print("\n[dim]No saved skills yet. Create one first![/dim]\n")
        return False

    table = Table(box=box.ROUNDED, border_style="cyan", header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Skill Name", width=28)
    table.add_column("Agent", width=16)
    table.add_column("Created", width=20)
    table.add_column("Prompt Preview", width=40)

    for i, skill in enumerate(skills, 1):
        created = skill.created_at[:19] if skill.created_at else "—"
        preview = skill.prompt_text[:37] + "…" if len(skill.prompt_text) > 37 else skill.prompt_text
        agent_label = AGENT_MARK.get(skill.agent_type, "•") + " " + skill.agent_type.title()
        table.add_row(str(i), skill.name, agent_label, created, preview)

    console.print()
    console.print(table)
    console.print()
    return True


def _list_skills(store: SkillsStore) -> None:
    """List all saved skills."""
    console.clear()
    console.print(make_header("Saved Skills"))
    _show_skills_table(store)
    Prompt.ask("[dim]Press Enter to go back[/dim]")


def _create_skill(store: SkillsStore) -> None:
    """Interactive skill creation."""
    console.clear()
    console.print(make_header("Create Skill"))
    console.print()

    # Skill name
    name = Prompt.ask("[bold cyan]Skill name[/bold cyan]")
    if not name.strip():
        console.print("[red]Skill name cannot be empty.[/red]")
        time.sleep(1)
        return

    # Check if name already exists
    existing = store.get_skill(name.strip())
    if existing:
        console.print(f"[red]A skill named '{name.strip()}' already exists.[/red]")
        time.sleep(1)
        return

    # Agent type selection
    all_agent_types = list(AGENT_MARK.keys())
    console.print("\n[bold]Target agent type:[/bold]")
    for i, agent in enumerate(all_agent_types, 1):
        console.print(f"  {i}. {AGENT_MARK.get(agent, '')} {agent.title()} — {AGENT_DESCRIPTIONS.get(agent, '')[:50]}")
    agent_choice = Prompt.ask(
        "[bold cyan]Select agent type[/bold cyan]",
        choices=[str(i) for i in range(1, len(all_agent_types) + 1)],
        default="1",
    )
    agent_type = all_agent_types[int(agent_choice) - 1]

    # Prompt text — multi-line
    console.print(f"\n[bold]Prompt text for {AGENT_MARK.get(agent_type, '')} {agent_type.title()}:[/bold]")
    console.print("[dim](Type your prompt, then enter '---' on a new line to finish)[/dim]")
    lines = []
    while True:
        line = input()
        if line.strip() == "---":
            break
        lines.append(line)
    prompt_text = "\n".join(lines).strip()

    if not prompt_text:
        console.print("[red]Prompt cannot be empty.[/red]")
        time.sleep(1)
        return

    try:
        store.add_skill(name.strip(), agent_type, prompt_text)
        console.print(f"\n[green]Skill '{name.strip()}' created for {agent_type.title()}[/green]")
    except ValueError as e:
        console.print(f"\n[red]{e}[/red]")

    time.sleep(1)


def _delete_skill(store: SkillsStore) -> None:
    """Interactive skill deletion."""
    console.clear()
    console.print(make_header("Delete Skill"))

    if not _show_skills_table(store):
        Prompt.ask("[dim]Press Enter to go back[/dim]")
        return

    name = Prompt.ask("[bold cyan]Name of skill to delete[/bold cyan]")
    if not name.strip():
        return

    if store.delete_skill(name.strip()):
        console.print(f"\n[green]Skill '{name.strip()}' deleted.[/green]")
    else:
        console.print(f"\n[red]No skill named '{name.strip()}' found.[/red]")

    time.sleep(1)


def _use_skill(store: SkillsStore, cfg: dict) -> None:
    """Pick a skill and run a workflow with it."""
    console.clear()
    console.print(make_header("Use a Skill"))

    if not _show_skills_table(store):
        Prompt.ask("[dim]Press Enter to go back[/dim]")
        return

    name = Prompt.ask("[bold cyan]Enter skill name to use[/bold cyan]")
    if not name.strip():
        return

    skill = store.get_skill(name.strip())
    if not skill:
        console.print(f"\n[red]No skill named '{name.strip()}' found.[/red]")
        time.sleep(1)
        return

    console.print(f"\n[green]Using skill:[/green] [bold]{skill.name}[/bold]")
    console.print(f"  Agent: {AGENT_MARK.get(skill.agent_type, '')} {skill.agent_type.title()}")
    console.print(f"  Prompt: {skill.prompt_text[:100]}")

    if not Confirm.ask("\n[bold cyan]Run this skill now?[/bold cyan]", default=True):
        return

    # Run the workflow with the skill as the goal
    _run_skill_workflow(skill, cfg)


def _run_skill_workflow(skill: Skill, cfg: dict) -> None:
    """Run a workflow centered on a saved skill."""
    from rich.markdown import Markdown

    console.clear()
    console.print(make_header("Running Skill Workflow", skill.name))
    console.print()

    store = TaskStore(db_path=cfg.get("db_path"))
    llm = LLMClient(
        api_key=cfg.get("api_key"),
        base_url=cfg.get("base_url"),
        model=cfg.get("model"),
    )
    planner = Planner(llm)
    tracker = Tracker(store)
    dispatcher = Dispatcher(
        task_store=store, llm_client=llm, tracker=tracker, parallel=cfg.get("parallel", False),
        custom_agents_dir=project_root,
    )

    # Plan from skill
    with console.status("[bold cyan]Planning subtasks from skill…[/bold cyan]", spinner="dots"):
        workflow = Workflow(
            goal=f"[SKILL: {skill.name}] {skill.prompt_text[:60]}",
            agents=cfg.get("agents", ["researcher", "writer", "reviewer"]),
            output_format=cfg.get("output_format", "markdown"),
        )
        store.save_workflow(workflow)
        subtasks = planner.plan_from_skill(
            skill_name=skill.name,
            prompt_text=skill.prompt_text,
            agent_type=skill.agent_type,
            agents=workflow.agents,
            workflow_id=workflow.id,
        )
        for st in subtasks:
            workflow.add_subtask(st)
            store.save_subtask(st)

    # Show plan
    plan_table = Table(box=box.SIMPLE, border_style="dim", padding=(0, 1))
    plan_table.add_column("Step", style="bold dim", width=5)
    plan_table.add_column("Agent", width=18)
    plan_table.add_column("Task", width=70)
    for i, st in enumerate(subtasks, 1):
        plan_table.add_row(str(i), str(agent_tag(st.agent_type)), st.description)
    console.print(Panel(plan_table, title="Skill Plan", border_style="cyan"))
    console.print()

    # Progress bar
    total_steps = len(subtasks)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    task_id = progress.add_task("[cyan]Executing…", total=total_steps)

    event_log: List[str] = []

    with progress:
        for i, st in enumerate(subtasks):
            agent_label = str(agent_tag(st.agent_type))
            desc_short = st.description[:60]

            progress.update(task_id, description=f"[cyan]{agent_label}: {desc_short}")

            try:
                result = dispatcher._execute_subtask_with_retry(st, {"previous_results": ""})
                store.save_subtask(st)
                status = store.get_subtask(st.id)
                status_val = status.status if status else SubtaskStatus.COMPLETED
                icon = "✓" if status_val == SubtaskStatus.COMPLETED else "✗"
            except Exception as e:
                result = f"Error: {e}"
                icon = "✗"

            event_log.append(f"  {icon} {agent_label}: {desc_short}")
            progress.update(task_id, advance=1)

        progress.update(task_id, description="[bold green]Workflow Complete![/bold green]")

    # Compile results
    final_output = dispatcher._compile_results(subtasks, {st.id: st.output_data or "" for st in subtasks})
    store.update_workflow_output(workflow.id, final_output)
    tracker.complete_workflow(workflow.id)

    console.print()
    console.print(Rule(style="cyan"))
    console.print("[bold cyan]Result[/bold cyan]")
    console.print()

    if cfg.get("output_format") == "json":
        try:
            data = json.loads(final_output)
            console.print_json(data=data)
        except Exception:
            console.print(final_output[:3000])
    else:
        md = Markdown(final_output[:3000])
        console.print(Panel(md, border_style="dim", title="Output", padding=(1, 2)))

    if len(final_output) > 3000:
        console.print("\n[dim]… (output truncated in preview)[/dim]")

    # Save option
    if Confirm.ask("\n[dim]Save output to file?[/dim]", default=False):
        fname = f"skill_{skill.name.replace(' ', '_')}_{workflow.id[:8]}.md"
        with open(os.path.join(project_root, fname), "w") as f:
            f.write(final_output)
        console.print(f"[green]✓ Saved to {fname}[/green]")

    console.print()
    console.print(Panel(
        f"[bold]Workflow ID:[/bold] {workflow.id}\n"
        f"[bold]Status:[/bold] {total_steps}/{total_steps} subtasks completed\n"
        f"[bold]Skill:[/bold] {skill.name} ({skill.agent_type.title()})",
        border_style="green",
        title="Done",
    ))
    Prompt.ask("\n[dim]Press Enter to continue[/dim]")


# ── Custom Agent Management ───────────────────────────────────────────────

def _get_custom_agent_store() -> CustomAgentStore:
    """Get a CustomAgentStore rooted at the project directory."""
    return CustomAgentStore(data_dir=project_root)


def show_create_agent(_cfg: dict) -> None:
    """Interactive flow to create a new custom agent."""
    store = _get_custom_agent_store()

    console.clear()
    console.print(make_header("Create Custom Agent"))
    console.print()

    # Step a: Name
    name = Prompt.ask("[bold cyan]Agent name[/bold cyan]")
    if not name.strip():
        console.print("[red]Agent name cannot be empty.[/red]")
        time.sleep(1)
        return
    clean_name = name.strip()

    # Check for duplicates
    if store.get(clean_name):
        console.print(f"[red]An agent named '{clean_name}' already exists.[/red]")
        time.sleep(1)
        return

    # Step b: System prompt (multi-line)
    console.print(f"\n[bold]System prompt for [bright_cyan]{clean_name}[/bright_cyan]:[/bold]")
    console.print("[dim](Describe what this agent does, its expertise, tone, etc. "
                  "Type '---' on a new line to finish)[/dim]")
    lines = []
    while True:
        line = input()
        if line.strip() == "---":
            break
        lines.append(line)
    system_prompt = "\n".join(lines).strip()

    if not system_prompt:
        console.print("[red]System prompt cannot be empty.[/red]")
        time.sleep(1)
        return

    # Step c: Confirmation
    console.print()
    console.print(Rule(style="cyan"))
    console.print(f"[bold]Agent:[/bold] {clean_name}")
    console.print(f"[bold]System prompt:[/bold]")
    console.print(Panel(system_prompt[:500], border_style="dim"))
    if len(system_prompt) > 500:
        console.print("[dim]… (truncated in preview)[/dim]")

    if not Confirm.ask("\n[bold cyan]Save this custom agent?[/bold cyan]", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        time.sleep(0.8)
        return

    try:
        store.add(clean_name, system_prompt)
        console.print(f"\n[green]Custom agent '{clean_name}' created![/green]")
        console.print("[dim]It is now available in Settings and Workflows.[/dim]")
    except ValueError as e:
        console.print(f"\n[red]{e}[/red]")

    time.sleep(1.5)


def show_manage_agents(_cfg: dict) -> None:
    """List and manage custom agents."""
    store = _get_custom_agent_store()

    while True:
        console.clear()
        console.print(make_header("Custom Agents"))
        console.print()

        agents = store.list()
        if not agents:
            console.print("[dim]No custom agents yet. Create one from the main menu![/dim]\n")
            Prompt.ask("\n[dim]Press Enter to go back[/dim]")
            return

        table = Table(box=box.ROUNDED, border_style="cyan", header_style="bold cyan")
        table.add_column("#", style="dim", width=3)
        table.add_column("Agent Name", width=28)
        table.add_column("Created", width=24)
        for i, agent in enumerate(agents, 1):
            created = agent.created_at[:19] if agent.created_at else "—"
            table.add_row(str(i), f"{agent.name}", created)
        console.print(table)
        console.print()

        menu = Table(box=box.SIMPLE, border_style="dim", show_header=False, padding=(0, 2))
        menu.add_column("Option", style="bold", width=4)
        menu.add_column("Action", width=40)
        menu.add_row("1", "Delete a custom agent")
        menu.add_row("2", "Back to Main Menu")
        console.print(menu)

        choice = Prompt.ask(
            "[bold cyan]Select option[/bold cyan]",
            choices=["1", "2"],
            default="2",
        )

        if choice == "1":
            _delete_custom_agent(store)
        elif choice == "2":
            break


def _delete_custom_agent(store: CustomAgentStore) -> None:
    """Delete a custom agent by name."""
    agents = store.list()
    if not agents:
        return

    name = Prompt.ask("[bold cyan]Name of agent to delete[/bold cyan]")
    if not name.strip():
        return

    if store.delete(name.strip()):
        console.print(f"\n[green]Agent '{name.strip()}' deleted.[/green]")
    else:
        console.print(f"\n[red]No custom agent named '{name.strip()}' found.[/red]")

    time.sleep(1)


# ── Main menu ─────────────────────────────────────────────────────────────────

def show_menu(cfg: dict) -> str:
    """Render the main menu and return the user's choice."""
    console.clear()

    # App header
    console.print(make_header("Agent Orchestrator", "Multi-Agent Workflow System"))
    console.print()

    # Config summary card
    agents_str = ", ".join(f"{AGENT_MARK.get(a, '')} {a.title()}" for a in cfg.get("agents", ["researcher", "writer", "reviewer"]))
    config_table = Table(box=box.SIMPLE, border_style="dim", padding=(0, 2))
    config_table.add_column("Setting", style="bold cyan", width=18)
    config_table.add_column("Value", width=50)
    config_table.add_row("Agents", agents_str)
    config_table.add_row("Format", cfg.get("output_format", "markdown").title())
    config_table.add_row("Parallel", "Yes" if cfg.get("parallel") else "No")

    # LLM status
    cfg_model = cfg.get("model", "") or os.environ.get("LLM_MODEL", "deepseek-chat")
    cfg_key = cfg.get("api_key", "") or ""
    env_key = os.environ.get("LLM_API_KEY", "")
    has_key = bool(cfg_key) or bool(env_key)
    key_display = "Set" if has_key else "Not set (use Settings or set LLM_API_KEY)"

    cfg_base = cfg.get("base_url", "") or ""
    provider_name = "Custom"
    PROVIDERS_INV = {
        "https://api.openai.com/v1": "OpenAI",
        "https://api.deepseek.com": "DeepSeek",
        "https://openrouter.ai/api/v1": "OpenRouter",
    }
    for purl, pname in PROVIDERS_INV.items():
        if cfg_base.rstrip("/") == purl.rstrip("/"):
            provider_name = pname
            break
    if not provider_name:
        provider_name = "Custom"

    config_table.add_row("Provider", provider_name)
    config_table.add_row("Model", cfg_model or "deepseek-chat")
    config_table.add_row("API Key", key_display)
    wa_configured = bool(os.environ.get("WHATSAPP_TOKEN") and os.environ.get("WHATSAPP_PHONE_NUMBER_ID"))
    config_table.add_row("WhatsApp", "Configured" if wa_configured else "Not set (WHATSAPP_TOKEN + WHATSAPP_PHONE_NUMBER_ID)")
    console.print(Panel(config_table, title="Current Settings", border_style="cyan"))
    console.print()

    # Menu options
    menu = Table(box=box.ROUNDED, border_style="cyan", show_header=False, padding=(0, 2))
    menu.add_column("Option", style="bold", width=4)
    menu.add_column("Action", width=50)
    menu.add_row("1", "New Workflow  —  Enter a goal and run it")
    menu.add_row("2", "History  —  View past workflow results")
    menu.add_row("3", "Settings  —  Configure agents, format, parallel mode")
    menu.add_row("4", "Skills  —  Manage reusable prompt skills")
    menu.add_row("5", "Create Agent  —  Define a new custom agent")
    menu.add_row("6", "Custom Agents  —  List & manage custom agents")
    menu.add_row("7", "Exit")
    console.print(menu)
    console.print()

    return Prompt.ask("[bold cyan]Select option[/bold cyan]", choices=[str(i) for i in range(1, 8)], default="1")


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    # Silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("src.llm").setLevel(logging.WARNING)

    while True:
        try:
            choice = show_menu(cfg)
            if choice == "1":
                show_goal_prompt(cfg)
            elif choice == "2":
                store = TaskStore(db_path=cfg.get("db_path"))
                view_history(store)
            elif choice == "3":
                cfg = edit_settings(cfg)
            elif choice == "4":
                show_skills_menu(cfg)
            elif choice == "5":
                show_create_agent(cfg)
            elif choice == "6":
                show_manage_agents(cfg)
            elif choice == "7":
                console.print("\n[cyan]Goodbye![/cyan]\n")
                break
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Exiting…[/dim]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            Prompt.ask("[dim]Press Enter to continue[/dim]")


if __name__ == "__main__":
    main()
