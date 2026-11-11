"""OpenAIProvider honors the configured embedding dimension via the `dimensions` param.

Critical for the gemini->OpenAI redirect: a gemini dataset's vector index is 768-dim, so the
OpenAI embeddings serving it must also be 768-dim or `embedding match` fails on the store."""

from __future__ import annotations

from types import SimpleNamespace

from lumintelligence.config import OpenAIConfig
from lumintelligence.llm import EmbedPurpose
from lumintelligence.llm.openai import OpenAIProvider


class _FakeEmbeddings:
    def __init__(self, sink):
        self._sink = sink

    def create(self, **kwargs):
        self._sink.append(kwargs)
        n = len(kwargs["input"])
        dim = kwargs.get("dimensions", 1536)
        return SimpleNamespace(
            usage=SimpleNamespace(total_tokens=1),
            data=[SimpleNamespace(embedding=[0.0] * dim) for _ in range(n)],
        )


def _provider(config):
    prov = OpenAIProvider(api_key="test-key", config=config)
    calls: list[dict] = []
    prov._client = SimpleNamespace(embeddings=_FakeEmbeddings(calls))
    return prov, calls


def test_text_embedding_3_passes_configured_dimensions():
    prov, calls = _provider(OpenAIConfig(embedding_model="text-embedding-3-small", embedding_dim=768))
    vecs = prov.embed(["a", "b"], EmbedPurpose.DOCUMENT)

    assert calls[0]["dimensions"] == 768
    assert len(vecs[0]) == 768
    assert prov.dimension == 768


def test_non_v3_model_omits_dimensions():
    prov, calls = _provider(OpenAIConfig(embedding_model="text-embedding-ada-002", embedding_dim=1536))
    prov.embed(["a"], EmbedPurpose.DOCUMENT)

    assert "dimensions" not in calls[0]   # ada-002 rejects the param
