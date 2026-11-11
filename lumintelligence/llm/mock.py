"""MockProvider — deterministic, no-network provider for tests and offline dry runs.

Supply ``json_handler(system_prompt, content, schema)`` to script LLM JSON responses
(e.g. a NoD stub that returns entities for given content). Embeddings are deterministic,
normalized pseudo-vectors derived from the text — stable across runs, but not
semantically meaningful (real cluster quality is validated in notebooks with a real
provider).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable

from lumintelligence.llm.base import EmbeddingProvider, EmbedPurpose, LLMProvider


class MockProvider(LLMProvider, EmbeddingProvider):
    def __init__(
        self,
        json_handler: Callable[[str, str, dict], object] | None = None,
        text_handler: Callable[[str, str], str] | None = None,
        embed_dim: int = 8,
    ):
        self._json_handler = json_handler
        self._text_handler = text_handler
        self._dim = embed_dim

    def complete_json(self, system_prompt: str, content: str, schema: dict) -> dict | list:
        if self._json_handler is not None:
            result = self._json_handler(system_prompt, content, schema)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return result
        return {}

    def complete(self, system_prompt: str, content: str) -> str:
        if self._text_handler is not None:
            return self._text_handler(system_prompt, content)
        return "mock response"

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str | None:
        return f"mock-{self._dim}"

    def embed(self, texts: list[str], purpose: EmbedPurpose = EmbedPurpose.DOCUMENT) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vals = [(digest[i % len(digest)] / 255.0) - 0.5 for i in range(self._dim)]
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / norm for v in vals]


def scripted_handler(system_prompt: str, content: str, schema: dict) -> object:
    """A generic scripted handler for offline runs of the whole engine and CLI (``LUM_PROVIDER=mock``):
    page filter keeps pages unless they look like a contents page; NoD returns capitalised word
    pairs; tagging returns one tag; the intention engine echoes those pairs; answers cite [1]."""
    import json
    import re

    props = schema.get("properties", {})
    if "has_content" in props:
        toc = "contents" in content.lower()
        return {"has_content": not toc, "reason": "table of contents" if toc else "body text"}
    if "answer" in props:
        n = re.search(r"^\[(\d+)\]", content, re.MULTILINE)
        return {"answer": f"Mock answer [{n.group(1)}]." if n else "No sources.",
                "cited": [int(n.group(1))] if n else []}
    names = list(dict.fromkeys(re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", content)))[:5]
    if "exact_keywords" in json.dumps(schema):
        return {"exact_keywords": names[:2], "expansion_keywords": []}
    if schema.get("additionalProperties") is False and "required" in schema:
        return {n: ["Named Person"] for n in schema["required"]}
    return {n: {"context": n, "inferred_type": "Person"} for n in names}
