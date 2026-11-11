"""OpenAIProvider.complete_json: strict json_schema for closed schemas, JSON-mode fallback on a
400, OutputTruncated on finish_reason == "length"."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lumintelligence.config import OpenAIConfig
from lumintelligence.llm.base import OutputTruncated, json_object_schema
from lumintelligence.llm.openai import OpenAIProvider

CLOSED = {"type": "object", "properties": {"A": {"type": "array", "items": {"type": "string"}}},
          "required": ["A"], "additionalProperties": False}


class _BadRequest(Exception):
    status_code = 400


class _FakeCompletions:
    def __init__(self, reject_schema=False, finish="stop", content='{"A": ["x"]}'):
        self.calls: list[dict] = []
        self.reject_schema, self.finish, self.content = reject_schema, finish, content

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_schema and kwargs["response_format"]["type"] == "json_schema":
            raise _BadRequest("Error code: 400 - response_format json_schema not supported")
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(
            finish_reason=self.finish, message=SimpleNamespace(content=self.content))])


def _provider(fake, **cfg):
    prov = OpenAIProvider(api_key="test-key", config=OpenAIConfig(**cfg))
    prov._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return prov


def test_closed_schema_is_sent_as_strict_json_schema():
    fake = _FakeCompletions()
    assert _provider(fake).complete_json("sys", "x", CLOSED) == {"A": ["x"]}
    fmt = fake.calls[0]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == CLOSED


def test_open_schema_uses_json_mode():
    fake = _FakeCompletions(content=json.dumps({"Acme": "ctx"}))
    _provider(fake).complete_json("sys", "x", json_object_schema("string"))
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_rejected_schema_falls_back_to_json_mode_for_the_session():
    fake = _FakeCompletions(reject_schema=True)
    prov = _provider(fake)
    assert prov.complete_json("sys", "x", CLOSED) == {"A": ["x"]}
    assert [c["response_format"]["type"] for c in fake.calls] == ["json_schema", "json_object"]
    prov.complete_json("sys", "x", CLOSED)
    assert fake.calls[-1]["response_format"]["type"] == "json_object"


def test_structured_output_can_be_switched_off():
    fake = _FakeCompletions()
    _provider(fake, structured_output=False).complete_json("sys", "x", CLOSED)
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_truncated_answer_raises():
    fake = _FakeCompletions(finish="length", content='{"A": ["x"')
    with pytest.raises(OutputTruncated):
        _provider(fake).complete_json("sys", "x", CLOSED)


def test_no_pause_by_default():
    assert OpenAIConfig().request_pause_s == 0.0
