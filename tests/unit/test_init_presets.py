"""Model modes and the guided `lum init`: presets, Ollama checks, skip, online key, prefixes."""

from __future__ import annotations

import json

import pytest

from lumintelligence import cli
from lumintelligence import home as H
from lumintelligence.llm.base import EmbedPurpose
from lumintelligence.llm.mock import MockProvider, scripted_handler
from lumintelligence.llm.openai import _task_prefix


def _json(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def offline_mock(monkeypatch):
    """Real readiness checks, mock model calls."""
    monkeypatch.delenv("LUM_PROVIDER", raising=False)
    mock = MockProvider(json_handler=scripted_handler, embed_dim=16)
    monkeypatch.setattr(H, "make_providers", lambda s, embedding_dim=None: (mock, mock))


def test_env_has_both_mode_sets_and_the_switch_note(capsys, monkeypatch):
    monkeypatch.setenv("LUM_PROVIDER", "mock")
    assert cli.main(["--json", "init", "--yes", "--no-skill"]) == 0
    capsys.readouterr()
    text = H.path(".env").read_text()
    assert "LUM_MODE=offline" in text and "kb rebuild" in text
    assert "# --- offline: Ollama ---" in text and "# --- online:" in text
    for mode in ("OFFLINE", "ONLINE"):
        for field in H.MODE_FIELDS:
            assert f"LUM_{mode}_{field}=" in text


def test_resolve_models_follows_mode_and_env_override(monkeypatch):
    s = {k: d for k, (d, _) in H.SETTINGS.items()}
    assert H.resolve_models(s)["chat_model"] == "hermes3"
    s["LUM_MODE"] = "online"
    r = H.resolve_models(s)
    assert (r["chat_model"], r["embed_model"], r["base_url"]) == ("gpt-6-luna", "text-embedding-3-small", "")
    s["LUM_MODE"] = "cloud"
    with pytest.raises(H.SetupError):
        H.resolve_models(s)


def test_missing_models_matches_latest_tag():
    assert H.missing_models(["hermes3:latest", "nomic-embed-text:latest"], ["hermes3", "nomic-embed-text"]) == []
    assert H.missing_models(["hermes3:8b"], ["hermes3", "nomic-embed-text"]) == ["hermes3", "nomic-embed-text"]


def test_offline_without_ollama_writes_env_but_no_kb(capsys, offline_mock, monkeypatch):
    monkeypatch.setattr(H, "ollama_status", lambda url, timeout=3.0: {"reachable": False, "models": []})
    assert cli.main(["--json", "init", "--yes", "--no-skill"]) == 0
    out = _json(capsys)
    assert out["ready"] is False and out["missing"] == ["Ollama"] and "ollama.com" in out["message"]
    assert H.path(".env").exists() and not H.path("kb.db").exists()


def test_missing_models_are_named_without_commands(capsys, offline_mock, monkeypatch):
    monkeypatch.setattr(H, "ollama_status", lambda url, timeout=3.0: {"reachable": True, "models": ["hermes3:latest"]})
    cli.main(["--json", "init", "--yes", "--no-skill"])
    out = _json(capsys)
    assert out["missing"] == ["nomic-embed-text"] and "pull" not in out["message"]


def test_interactive_recheck_then_ready(capsys, offline_mock, monkeypatch):
    states = iter([{"reachable": False, "models": []},
                   {"reachable": True, "models": ["hermes3:latest"]},
                   {"reachable": True, "models": ["hermes3:latest", "nomic-embed-text:latest"]}])
    monkeypatch.setattr(H, "ollama_status", lambda url, timeout=3.0: next(states))
    monkeypatch.setattr(cli, "_tty_in", lambda: True)
    answers = iter(["", "", "", "", "", "n"])   # mode, chat model, embed model, re-check x2, skill: no
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    cli.main(["init"])
    out = capsys.readouterr().out
    assert "Ollama is not running" in out and "Install these models in Ollama: nomic-embed-text" in out
    assert '"ready": true' in out and H.path("kb.db").exists()   # summary is JSON: stdout isn't a tty here


def test_interactive_skip_ends_without_kb(capsys, offline_mock, monkeypatch):
    monkeypatch.setattr(H, "ollama_status", lambda url, timeout=3.0: {"reachable": False, "models": []})
    monkeypatch.setattr(cli, "_tty_in", lambda: True)
    answers = iter(["offline", "", "", "skip", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    cli.main(["init"])
    out = capsys.readouterr().out
    assert '"ready": false' in out and '"skipped": true' in out
    assert not H.path("kb.db").exists() and H.read_env()["LUM_MODE"] == "offline"


def test_online_asks_for_a_hidden_key(capsys, offline_mock, monkeypatch):
    monkeypatch.setattr(cli, "_tty_in", lambda: True)
    monkeypatch.setattr("getpass.getpass", lambda _: "sk-test")
    answers = iter(["online", "", "", "", "n"])     # mode, base URL, chat, embed, skill: no
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    cli.main(["init"])
    capsys.readouterr()
    env = H.read_env()
    assert env["LUM_MODE"] == "online" and env["LUM_ONLINE_API_KEY"] == "sk-test"
    assert env["LUM_OFFLINE_CHAT_MODEL"] == "hermes3"          # the other set stays ready


def test_online_without_key_is_not_ready(capsys, offline_mock):
    cli.main(["--json", "init", "--yes", "--mode", "online", "--no-skill"])
    out = _json(capsys)
    assert out["ready"] is False and "LUM_ONLINE_API_KEY" in out["message"]


def test_nomic_task_prefixes():
    assert _task_prefix("nomic-embed-text", EmbedPurpose.QUERY) == "search_query: "
    assert _task_prefix("nomic-embed-text", EmbedPurpose.DOCUMENT) == "search_document: "
    assert _task_prefix("nomic-embed-text:v1.5", EmbedPurpose.CLUSTERING) == "clustering: "
    assert _task_prefix("text-embedding-3-small", EmbedPurpose.QUERY) == ""
