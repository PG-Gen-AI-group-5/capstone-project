"""Versioned prompt files (task 11.4): src/llm/prompts/<agent>/v<N>.md.

A file has two sections, `## System` and `## User`. The user section is a template: `$name` is replaced by
the JSON of variables[name]. The version used is recorded with every output and in the run manifest.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

PROMPT_DIR = Path(__file__).with_name("prompts")


@dataclass(frozen=True)
class Prompt:
    agent: str
    version: str
    system: str
    user: str

    def render(self, variables: dict) -> str:
        values = {k: v if isinstance(v, str) else json.dumps(v, default=str, indent=1) for k, v in variables.items()}
        return Template(self.user).safe_substitute(values)


def versions(agent: str) -> list[str]:
    files = sorted((PROMPT_DIR / agent).glob("v*.md"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    return [p.stem for p in files]


def load_prompt(agent: str, version: str | None = None) -> Prompt:
    """The given version of an agent's prompt, or its latest."""
    available = versions(agent)
    if not available:
        raise FileNotFoundError(f"no prompt files in {PROMPT_DIR / agent}")
    version = version or available[-1]
    text = (PROMPT_DIR / agent / f"{version}.md").read_text(encoding="utf-8")
    parts = re.split(r"^## (System|User)\s*$", text, flags=re.MULTILINE)
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    return Prompt(agent, version, sections.get("System", ""), sections.get("User", ""))
