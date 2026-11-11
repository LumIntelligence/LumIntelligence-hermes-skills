"""Provider registry — select a provider by name, and let users add their own.

"openai" is registered by default and covers OpenAI plus any OpenAI-compatible endpoint via
``base_url`` (local runtimes). Any class implementing LLMProvider + EmbeddingProvider can
be registered and selected by name — no library changes needed:

    from lumintelligence.llm import register_provider, make_provider
    register_provider("mine", lambda **kw: MyProvider(**kw))
    provider = make_provider("mine", llm_model="...", api_key="...")

A factory receives keyword args (llm_model, embedding_model, embedding_dim, api_key, base_url, ...)
and returns a ready provider. Unknown kwargs are ignored, so callers can pass a superset.
"""

from __future__ import annotations

from collections.abc import Callable

from lumintelligence.config import OpenAIConfig

_REGISTRY: dict[str, Callable] = {}


def register_provider(name: str, factory: Callable) -> None:
    _REGISTRY[name.lower()] = factory


def make_provider(name: str, **kwargs):
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown provider '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def _openai_factory(llm_model="gpt-6-luna", embedding_model="text-embedding-3-small",
                    embedding_dim=1536, api_key=None, base_url=None, reasoning_effort="none",
                    request_pause_s=0.0, on_tokens=None, **_):
    from lumintelligence.llm.openai import OpenAIProvider
    cfg = OpenAIConfig(llm_model=llm_model, embedding_model=embedding_model, embedding_dim=embedding_dim,
                       base_url=base_url, reasoning_effort=reasoning_effort, request_pause_s=request_pause_s)
    return OpenAIProvider(api_key=api_key, config=cfg, on_tokens=on_tokens)


register_provider("openai", _openai_factory)


def _mock_factory(**_):
    """Offline provider with a fixed 16-dimension embedder (like a real model, it ignores settings)."""
    from lumintelligence.llm.mock import MockProvider, scripted_handler
    return MockProvider(json_handler=scripted_handler, embed_dim=16)


register_provider("mock", _mock_factory)
