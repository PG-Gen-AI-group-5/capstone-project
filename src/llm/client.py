"""LLM client (tasks 11.1-11.4): every agent calls the LLM through `LLMClient.complete`.

- provider `offline` (the default until task 1.10 picks a model): the agent's rule-based stand-in
  produces the output, so the whole system runs with no API key. Traces record provider "offline".
- provider `openai_compatible`: any OpenAI-style Chat Completions endpoint (NVIDIA NIM, OpenRouter, a local
  server), configured from JFF_LLM_* variables in .env (see openai_compatible.py).
- provider `anthropic`: Claude through the official SDK, JSON output constrained to the agent's
  pydantic schema (`output_config.format`) and validated again on return. Current Claude models reject
  `temperature`; repeatability comes from the response cache instead.
- A reply that doesn't fit the agent's schema is retried once with the validation error; if it still
  doesn't fit, the agent's offline rules produce the output (and the trace says so).
- Every response is cached by provider + model + effort + prompt version + rendered prompt + schema,
  so a re-run costs nothing (11.2). Each call is appended to a cost log; a daily budget in dollars and a
  daily call limit stop LLM calls once reached (11.3). Prompts are versioned files (11.4, see promptlib.py).
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import BaseModel, ValidationError

from src.config import Config
from src.llm.openai_compatible import call_chat_completions, endpoint_settings
from src.llm.promptlib import Prompt, load_prompt

DEFAULTS = {
    "provider": "offline", "model": "claude-opus-5-5", "effort": "medium", "max_tokens": 16000,
    "timeout_sec": 120, "max_retries": 2, "daily_budget_usd": 5.0, "daily_call_limit": 500,
    "price_per_mtok": {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0},
}


class BudgetExceeded(RuntimeError):
    pass


class LLMRefusal(RuntimeError):
    pass


@dataclass
class LLMResult:
    output: BaseModel
    provider: str
    prompt_version: str
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    note: str = ""


def strict_schema(model: type[BaseModel]) -> dict:
    """JSON schema for structured output: every object closed and every property required. Numeric
    bounds and defaults are left to pydantic's own validation after the call."""
    schema = model.model_json_schema()

    def fix(node):
        if isinstance(node, dict):
            for k in ("default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                node.pop(k, None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)
    fix(schema)
    return schema


class LLMClient:
    def __init__(self, cfg: Config, tracer=None):
        self.settings = {**DEFAULTS, **(cfg.get("llm") or {})}
        self.provider = self.settings["provider"]
        if self.provider == "openai_compatible":            # the model is named in .env (JFF_LLM_MODEL)
            self.settings["model"] = endpoint_settings(self.settings)["model"]
        self.cache_dir = cfg.data_dir / "cache" / "llm"
        self.cost_log = cfg.data_dir / "cache" / "llm_costs.jsonl"
        self.tracer = tracer
        self.prompt_versions = self.settings.get("prompt_versions") or {}

    # ---- public ----
    def complete(self, agent: str, variables: dict, schema: type[BaseModel], fallback: Callable[[], BaseModel],
                 symbol: str | None = None) -> LLMResult:
        prompt = load_prompt(agent, self.prompt_versions.get(agent))
        if self.provider == "offline":
            result = LLMResult(fallback(), "offline", prompt.version, note="offline rules")
        else:
            result = self._call(prompt, variables, schema, fallback)
        if self.tracer:
            self.tracer.step(agent, "offline" if result.provider == "offline" else "llm_call", symbol=symbol,
                             prompt_version=result.prompt_version, provider=result.provider,
                             input_summary=_short(variables), output=result.output.model_dump(mode="json"),
                             input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                             cost_usd=result.cost_usd, latency_ms=result.latency_ms, cached=result.cached)
        return result

    def spent_today(self) -> float:
        return sum(r["cost_usd"] for r in self._today())

    def calls_today(self) -> int:
        return len(self._today())

    def _today(self) -> list[dict]:
        if not self.cost_log.exists():
            return []
        today = datetime.now(timezone.utc).date().isoformat()
        rows = (json.loads(line) for line in self.cost_log.read_text(encoding="utf-8").splitlines())
        return [r for r in rows if r["at"][:10] == today]

    # ---- internals ----
    def _key(self, prompt: Prompt, user: str, schema: dict) -> str:
        s = self.settings
        blob = json.dumps([self.provider, s["model"], s["effort"], prompt.agent, prompt.version, prompt.system, user,
                           schema], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _call(self, prompt: Prompt, variables: dict, schema: type[BaseModel], fallback) -> LLMResult:
        user = prompt.render(variables)
        js = strict_schema(schema)
        key = self._key(prompt, user, js)
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            return LLMResult(schema.model_validate_json(path.read_text(encoding="utf-8")), self.provider,
                             prompt.version, cached=True)
        if self.spent_today() >= float(self.settings["daily_budget_usd"]):
            return LLMResult(fallback(), "offline", prompt.version, note="daily LLM budget spent; offline rules")
        if self.calls_today() >= int(self.settings["daily_call_limit"]):
            return LLMResult(fallback(), "offline", prompt.version, note="daily LLM call limit reached; offline rules")
        t0 = time.time()
        call = PROVIDERS[self.provider]
        totals = {"input_tokens": 0, "output_tokens": 0}
        output, ask = None, user
        for _attempt in range(2):                        # one retry with the validation error
            try:
                text, usage = call(self.settings, prompt.system, ask, js)
            except (LLMRefusal, ValueError) as e:
                return LLMResult(fallback(), "offline", prompt.version, note=f"LLM call failed ({e}); offline rules")
            self._log(prompt.agent, usage)
            for k in totals:
                totals[k] += usage.get(k, 0)
            try:
                output = schema.model_validate_json(text)
                break
            except ValidationError as e:
                first = e.errors()[0]
                ask = (f"{user}\n\nYour previous reply did not match the schema ({e.error_count()} error(s); first: "
                       f"{'.'.join(map(str, first['loc']))}: {first['msg']}). Reply again with only the corrected "
                       "JSON object.")
        if output is None:
            return LLMResult(fallback(), "offline", prompt.version, input_tokens=totals["input_tokens"],
                             output_tokens=totals["output_tokens"],
                             note="LLM reply did not fit the schema twice; offline rules")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output.model_dump_json(), encoding="utf-8")
        return LLMResult(output, self.provider, prompt.version, input_tokens=totals["input_tokens"],
                         output_tokens=totals["output_tokens"], cost_usd=self._cost(totals),
                         latency_ms=int((time.time() - t0) * 1000))

    def _log(self, agent: str, usage: dict) -> None:
        self.cost_log.parent.mkdir(parents=True, exist_ok=True)
        with self.cost_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "agent": agent, "provider": self.provider,
                                "model": self.settings["model"], **usage, "cost_usd": self._cost(usage)}) + "\n")

    def _cost(self, usage: dict) -> float:
        prices = (self.settings.get(self.provider) or {}).get("price_per_mtok") or self.settings["price_per_mtok"]
        p = {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0, **prices}
        return round((usage.get("input_tokens", 0) * p["input"] + usage.get("output_tokens", 0) * p["output"]
                      + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
                      + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]) / 1e6, 6)


def call_anthropic(settings: dict, system: str, user: str, schema: dict) -> tuple[str, dict]:
    """One Claude call with JSON output constrained to `schema`. Refusals raise LLMRefusal; the
    server-side fallback re-runs a declined request on Anthropic's recommended fallback model."""
    import anthropic

    client = anthropic.Anthropic(timeout=float(settings["timeout_sec"]), max_retries=int(settings["max_retries"]))
    response = client.beta.messages.create(
        model=settings["model"],
        max_tokens=int(settings["max_tokens"]),
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_config={"effort": settings["effort"], "format": {"type": "json_schema", "schema": schema}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise LLMRefusal(getattr(response.stop_details, "category", None) or "refused")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise ValueError(f"no text block (stop_reason={response.stop_reason})")
    u = response.usage
    usage = {k: int(getattr(u, k, 0) or 0) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                       "cache_creation_input_tokens")}
    return text, usage


PROVIDERS: dict[str, Callable[[dict, str, str, dict], tuple[str, dict]]] = {
    "anthropic": call_anthropic, "openai_compatible": call_chat_completions}


def _short(variables: dict, limit: int = 300) -> str:
    text = json.dumps(variables, default=str)
    return text if len(text) <= limit else text[:limit] + "..."

