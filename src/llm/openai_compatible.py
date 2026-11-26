"""Provider `openai_compatible`: any OpenAI-style Chat Completions endpoint (NVIDIA NIM, OpenRouter, a local
vLLM or Ollama server). Ported from the job-form-filler backend (app/tailoring/config.py, extractor.py).

Settings come from environment variables (put them in .env), named in `llm.openai_compatible` in config.yaml:
    JFF_LLM_BASE_URL   e.g. https://integrate.api.nvidia.com/v1  (HTTPS; plain HTTP only for localhost)
    JFF_LLM_API_KEY    sent as a Bearer token; never logged
    JFF_LLM_MODEL      e.g. nvidia/nemotron-3-ultra-550b-a55b
    JFF_LLM_TIMEOUT    seconds (default 120)

These endpoints can't constrain output to a schema, so the agent's JSON schema goes into the system message
and the reply is parsed as JSON (a Markdown code fence around it is removed). The client validates it with
pydantic and asks once more, with the error, when it doesn't fit.
"""
from __future__ import annotations

import json
import os
import re
from urllib.parse import urlsplit, urlunsplit

LLM_ENDPOINT_GUIDANCE = ("The configured LLM endpoint is invalid. Use HTTPS for remote providers; "
                         "HTTP is allowed only for localhost or loopback.")
_HTTP_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
ENV_DEFAULTS = {"base_url_env": "JFF_LLM_BASE_URL", "api_key_env": "JFF_LLM_API_KEY", "model_env": "JFF_LLM_MODEL",
                "timeout_env": "JFF_LLM_TIMEOUT", "temperature": 0.0, "json_mode": False}
_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", re.IGNORECASE | re.DOTALL)


class LLMConfigurationError(ValueError):
    """Safe configuration error that never includes the configured URL or key."""


def chat_completions_url(base_url: str) -> str:
    """Validate the configured endpoint and return its Chat Completions URL."""
    raw = base_url.strip() if isinstance(base_url, str) else ""
    try:
        parts = urlsplit(raw)
        hostname = parts.hostname
        _ = parts.port  # invalid or out-of-range ports fail here
    except ValueError:
        raise LLMConfigurationError(LLM_ENDPOINT_GUIDANCE) from None
    if (not raw or any(ch.isspace() for ch in raw) or not hostname or parts.username is not None
            or parts.password is not None or parts.fragment):
        raise LLMConfigurationError(LLM_ENDPOINT_GUIDANCE)
    scheme, hostname = parts.scheme.casefold(), hostname.casefold()
    if scheme != "https" and not (scheme == "http" and hostname in _HTTP_LOOPBACK_HOSTS):
        raise LLMConfigurationError(LLM_ENDPOINT_GUIDANCE)
    path = parts.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path += "/chat/completions"
    return urlunsplit((scheme, parts.netloc, path, parts.query, ""))


def endpoint_settings(settings: dict) -> dict:
    """Base URL, key, model and timeout from the environment variables named in the settings."""
    s = {**ENV_DEFAULTS, **(settings.get("openai_compatible") or {})}
    env = lambda name: os.environ.get(s[name], "").strip()
    return {"base_url": env("base_url_env"), "api_key": env("api_key_env"),
            "model": env("model_env") or str(settings.get("model") or ""),
            "timeout": float(env("timeout_env") or settings.get("timeout_sec") or 120),
            "temperature": float(s["temperature"]), "json_mode": bool(s["json_mode"])}


def configuration_ready(settings: dict) -> bool:
    e = endpoint_settings(settings)
    if not (e["base_url"] and e["api_key"] and e["model"]):
        return False
    try:
        chat_completions_url(e["base_url"])
    except LLMConfigurationError:
        return False
    return True


def schema_instructions(schema: dict) -> str:
    return ("\n\nReply with one JSON object only: no prose before or after it, no Markdown. It must match this "
            "JSON schema exactly (every property present, no extra properties):\n" + json.dumps(schema, indent=1))


def extract_json(content: str) -> str:
    """The JSON object in a reply: inside a ```json fence if there is one, else from the first '{' to the last '}'."""
    text = (content or "").strip()
    fenced = _FENCE.fullmatch(text) or _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in the reply")
        text = text[start:end + 1]
    json.loads(text)                       # raises ValueError (JSONDecodeError) when it isn't JSON
    return text


def call_chat_completions(settings: dict, system: str, user: str, schema: dict) -> tuple[str, dict]:
    """One Chat Completions call; returns (JSON text, token usage). Any failure raises ValueError, so the
    caller falls back to the agent's rules."""
    import httpx

    e = endpoint_settings(settings)
    if not (e["base_url"] and e["api_key"] and e["model"]):
        raise LLMConfigurationError("JFF_LLM_BASE_URL, JFF_LLM_API_KEY and JFF_LLM_MODEL must be set in .env")
    payload = {"model": e["model"], "temperature": e["temperature"], "max_tokens": int(settings.get("max_tokens", 4096)),
               "messages": [{"role": "system", "content": system + schema_instructions(schema)},
                            {"role": "user", "content": user}]}
    if e["json_mode"]:
        payload["response_format"] = {"type": "json_object"}
    if "nemotron" in e["model"].lower():
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        with httpx.Client(timeout=httpx.Timeout(e["timeout"]), follow_redirects=False) as client:
            resp = client.post(chat_completions_url(e["base_url"]), headers={"Authorization": f"Bearer {e['api_key']}"},
                               json=payload)
    except httpx.HTTPError as exc:
        raise ValueError(f"LLM request failed: {type(exc).__name__}") from None
    if resp.status_code != 200:
        raise ValueError(f"LLM endpoint returned status {resp.status_code}")
    data = resp.json()
    choices = data.get("choices")
    if not choices or not isinstance(choices, list):
        raise ValueError("LLM reply has no choices")
    content = (choices[0].get("message") or {}).get("content") or ""
    u = data.get("usage") or {}
    usage = {"input_tokens": int(u.get("prompt_tokens") or 0), "output_tokens": int(u.get("completion_tokens") or 0)}
    return extract_json(content), usage
