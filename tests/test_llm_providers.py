import copy
import json

import httpx
import pytest

from src.config import Config
from src.contracts import PlannerOutput, StockTask
from src.llm import client as llm_client
from src.llm.client import LLMClient
from src.llm.openai_compatible import (
    LLMConfigurationError,
    call_chat_completions,
    chat_completions_url,
    extract_json,
)

ENV = {"JFF_LLM_BASE_URL": "https://llm.example.com/v1", "JFF_LLM_API_KEY": "k", "JFF_LLM_MODEL": "vendor/nemotron-x",
       "JFF_LLM_TIMEOUT": "5"}


def test_endpoint_must_be_https_except_on_localhost():
    assert chat_completions_url("https://integrate.api.nvidia.com/v1") == \
        "https://integrate.api.nvidia.com/v1/chat/completions"
    assert chat_completions_url("http://localhost:8000/v1/") == "http://localhost:8000/v1/chat/completions"
    for bad in ("http://example.com/v1", "https://user:pw@example.com/v1", "", "https://exa mple.com"):
        with pytest.raises(LLMConfigurationError):
            chat_completions_url(bad)


def test_json_is_taken_from_a_fence_or_from_around_prose():
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Here it is: {"a": {"b": 2}} done') == '{"a": {"b": 2}}'
    with pytest.raises(ValueError):
        extract_json("no json here")


def mock_client(monkeypatch, handler):
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def test_the_request_carries_the_key_the_schema_and_no_thinking_for_nemotron(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    seen = {}

    def handler(request):
        seen.update(url=str(request.url), auth=request.headers["authorization"], body=json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '```json\n{"a": 1}\n```'}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    mock_client(monkeypatch, handler)
    text, usage = call_chat_completions({"max_tokens": 100}, "system text", "user text", {"type": "object"})
    assert text == '{"a": 1}' and usage == {"input_tokens": 10, "output_tokens": 5}
    assert seen["url"] == "https://llm.example.com/v1/chat/completions" and seen["auth"] == "Bearer k"
    body = seen["body"]
    assert body["model"] == "vendor/nemotron-x" and body["temperature"] == 0.0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "JSON schema" in body["messages"][0]["content"] and body["messages"][1]["content"] == "user text"


def test_errors_become_value_errors_so_the_agent_falls_back(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    mock_client(monkeypatch, lambda request: httpx.Response(429, json={}))
    with pytest.raises(ValueError, match="status 429"):
        call_chat_completions({}, "s", "u", {})
    monkeypatch.delenv("JFF_LLM_API_KEY")
    with pytest.raises(ValueError):
        call_chat_completions({}, "s", "u", {})


def test_a_reply_that_misses_the_schema_is_retried_once_with_the_error(tmp_path, monkeypatch):
    raw = copy.deepcopy(Config.load().raw)
    raw["paths"]["data"] = "data"
    raw["llm"] = {"provider": "fake"}
    cfg = Config(raw, tmp_path)
    asked = []
    replies = iter(['{"selected": "not a list", "skipped": 0, "notes": ""}',
                    '{"selected": [{"symbol": "AAA", "reasons": ["r"], "setup_type": "none"}], "skipped": 0, "notes": ""}'])

    def fake(settings, system, user, schema):
        asked.append(user)
        return next(replies), {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setitem(llm_client.PROVIDERS, "fake", fake)
    out = LLMClient(cfg).complete("planner", {"trade_date": "2024-06-03", "max_stocks": 5, "candidates": []},
                                  PlannerOutput, lambda: PlannerOutput(selected=[StockTask(symbol="ZZZ", reasons=[])],
                                                                       skipped=0))
    assert out.output.selected[0].symbol == "AAA" and out.provider == "fake"
    assert len(asked) == 2 and "did not match the schema" in asked[1]
