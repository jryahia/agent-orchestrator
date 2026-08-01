"""JSON-based Prompt Skills storage module.

A "Prompt Skill" is a saved reusable prompt that can target a specific agent.
Skills are persisted in a .orchestrator_skills.json file in the project root.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class Skill:
    """A reusable prompt skill targeting a specific agent type."""

    name: str
    agent_type: str
    prompt_text: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SkillsStore:
    """Loads, saves, lists, and deletes skills from a JSON file."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def _load(self) -> Dict[str, Any]:
        """Load the skills JSON file. Returns an empty dict if missing or corrupt."""
        if not os.path.exists(self.file_path):
            return {"skills": []}
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"skills": []}

    def _save(self, data: Dict[str, Any]) -> None:
        """Write skills data to the JSON file."""
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_skill(self, name: str, agent_type: str, prompt: str) -> Skill:
        """Add a new skill and persist it. Raises ValueError if name already exists."""
        data = self._load()
        existing_names = [s["name"] for s in data["skills"]]
        if name in existing_names:
            raise ValueError(f"Skill '{name}' already exists.")

        skill = Skill(name=name, agent_type=agent_type, prompt_text=prompt)
        data["skills"].append(asdict(skill))
        self._save(data)
        return skill

    def list_skills(self) -> List[Skill]:
        """Return all saved skills."""
        data = self._load()
        return [Skill(**s) for s in data["skills"]]

    def get_skill(self, name: str) -> Optional[Skill]:
        """Look up a skill by name. Returns None if not found."""
        data = self._load()
        for s in data["skills"]:
            if s["name"] == name:
                return Skill(**s)
        return None

    def delete_skill(self, name: str) -> bool:
        """Delete a skill by name. Returns True if deleted, False if not found."""
        data = self._load()
        before = len(data["skills"])
        data["skills"] = [s for s in data["skills"] if s["name"] != name]
        if len(data["skills"]) == before:
            return False
        self._save(data)
        return True
