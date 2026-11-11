"""Provider interfaces (ABCs) + small JSON-schema helpers.

Pipeline stages depend only on these abstractions (dependency injection), so OpenAI, local
OpenAI-compatible servers, and the test mock are interchangeable. ``EmbedPurpose`` is the
provider-neutral name for the embedding job; a provider may map it to its own mechanism
(e.g. a task type) or ignore it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class EmbedPurpose(str, Enum):
    CLUSTERING = "clustering"   # concept consolidation similarity
    DOCUMENT = "document"       # vector-store records
    QUERY = "query"             # retrieval query (deferred this iteration)


class OutputTruncated(RuntimeError):
    """The server stopped the answer at its length limit, so the JSON is incomplete. Callers
    that batch (tagging) split the batch and retry; others treat it as a failed attempt."""


class LLMProvider(ABC):
    @abstractmethod
    def complete_json(self, system_prompt: str, content: str, schema: dict) -> dict | list:
        """Return parsed JSON conforming to ``schema``."""

    @abstractmethod
    def complete(self, system_prompt: str, content: str) -> str:
        """Return a plain-text completion (used by the deferred synthesis step)."""


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""

    @abstractmethod
    def embed(self, texts: list[str], purpose: EmbedPurpose) -> list[list[float]]:
        """Embed a batch of texts for the given purpose."""

    @property
    def model_id(self) -> str | None:
        """Identifies the embedding model, so a vector index can tell when it changes. ``None``
        means unknown (no model check)."""
        return None

    def embed_one(self, text: str, purpose: EmbedPurpose) -> list[float]:
        return self.embed([text], purpose)[0]


# --- JSON-schema helpers (match the response shapes the pipeline needs) ----- #
def json_object_schema(value_type: str = "string") -> dict:
    """Object with dynamic string keys → typed values. NoD returns {name: context}."""
    return {"type": "object", "additionalProperties": {"type": value_type}}


def json_array_schema(item_type: str = "string") -> dict:
    """Array of typed items. ADHD returns [tag, ...]."""
    return {"type": "array", "items": {"type": item_type}}
