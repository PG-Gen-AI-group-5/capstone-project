"""Trace writer (tasks 14.1-14.3): every LLM call, offline rule output, debate turn, critique and decision of
a run becomes one TraceStep line in runs/<run_id>/trace.jsonl. The run manifest records what produced it."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.config import Config
from src.contracts import TraceStep
from src.store.db import config_hash, git_commit


class TraceWriter:
    def __init__(self, run_dir: Path, run_id: str):
        self.run_id = run_id
        self.path = Path(run_dir) / "trace.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._step = sum(1 for _ in self.path.open(encoding="utf-8")) if self.path.exists() else 0
        self._lock = threading.Lock()

    def step(self, agent: str, kind: str, **fields) -> TraceStep:
        with self._lock:
            self._step += 1
            row = TraceStep(run_id=self.run_id, step=self._step, at=datetime.now(timezone.utc), agent=agent,
                            kind=kind, **fields)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(row.model_dump_json() + "\n")
        return row


def read_trace(path: Path) -> list[dict]:
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def write_run_manifest(cfg: Config, run_dir: Path, run_id: str, **info) -> Path:
    """Task 14.3: git commit, config hash, prompt versions, LLM provider/model/cutoff, plus run details."""
    from src.llm.promptlib import versions

    llm = cfg.get("llm") or {}
    model = llm.get("model")
    if llm.get("provider") == "openai_compatible":
        from src.llm.openai_compatible import endpoint_settings
        model = endpoint_settings(llm)["model"]
    agents = ["planner", "retriever", "bull", "bear", "forecaster", "critic", "exit_monitor"]
    pinned = llm.get("prompt_versions") or {}
    manifest = {
        "run_id": run_id, "git_commit": git_commit(), "built_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(cfg.raw),
        "llm": {"provider": llm.get("provider", "offline"), "model": model,
                "training_cutoff": llm.get("training_cutoff")},
        "prompt_versions": {a: pinned.get(a) or (versions(a) or [None])[-1] for a in agents},
        **info,
    }
    path = Path(run_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
