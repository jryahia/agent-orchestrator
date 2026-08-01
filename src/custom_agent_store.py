"""CustomAgentStore — persistent JSON store for user-defined custom agents."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


CUSTOM_AGENTS_FILE = ".orchestrator_custom_agents.json"


@dataclass
class CustomAgentDef:
    """Definition of a custom agent stored on disk."""
    name: str
    system_prompt: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CustomAgentStore:
    """Manages persistent storage of custom agent definitions.

    Agents are stored in a JSON file as a list of objects with
    name, system_prompt, and created_at fields.
    """

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the store.

        Args:
            data_dir: Directory to store the JSON file in.
                      Defaults to the current working directory.
        """
        if data_dir is None:
            data_dir = os.getcwd()
        self._path = os.path.join(data_dir, CUSTOM_AGENTS_FILE)
        self._agents: Dict[str, CustomAgentDef] = {}
        self._load()

    # ── Public API ──────────────────────────────────────────────────────────

    def add(self, name: str, system_prompt: str) -> CustomAgentDef:
        """Add a new custom agent definition."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Agent name cannot be empty.")
        if not system_prompt.strip():
            raise ValueError("System prompt cannot be empty.")

        agent = CustomAgentDef(
            name=clean_name,
            system_prompt=system_prompt.strip(),
        )
        self._agents[clean_name] = agent
        self._save()
        return agent

    def get(self, name: str) -> Optional[CustomAgentDef]:
        """Get a custom agent by name. Returns None if not found."""
        return self._agents.get(name)

    def list(self) -> List[CustomAgentDef]:
        """Return all registered custom agents sorted by name."""
        return sorted(self._agents.values(), key=lambda a: a.name)

    def delete(self, name: str) -> bool:
        """Delete a custom agent by name. Returns True if deleted."""
        if name in self._agents:
            del self._agents[name]
            self._save()
            return True
        return False

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return a dict of agent_name -> {system_prompt, agent_type}.

        This is used by the Dispatcher to create CustomAgent instances
        on-the-fly. The agent_type is always 'custom'.
        """
        return {
            name: {"system_prompt": defn.system_prompt, "agent_type": "custom"}
            for name, defn in self._agents.items()
        }

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load agents from the JSON file."""
        if not os.path.exists(self._path):
            self._agents = {}
            return
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # New format: name -> definition dict
                for name, entry in data.items():
                    self._agents[name] = CustomAgentDef(
                        name=entry.get("name", name),
                        system_prompt=entry["system_prompt"],
                        created_at=entry.get("created_at", ""),
                    )
            elif isinstance(data, list):
                # Legacy format: list of definitions
                for entry in data:
                    name = entry["name"]
                    self._agents[name] = CustomAgentDef(
                        name=name,
                        system_prompt=entry["system_prompt"],
                        created_at=entry.get("created_at", ""),
                    )
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            self._agents = {}

    def _save(self) -> None:
        """Write all agents to the JSON file."""
        data = {
            name: asdict(defn) for name, defn in self._agents.items()
        }
        dirname = os.path.dirname(self._path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)
