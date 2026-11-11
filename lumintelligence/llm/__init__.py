"""Provider abstraction (Step 2).

LLMProvider / EmbeddingProvider ABCs + EmbedPurpose, schema helpers, and the concrete
providers: OpenAIProvider (OpenAI or any OpenAI-compatible endpoint via ``base_url``; the
``openai`` SDK is imported only when one is constructed) and MockProvider (deterministic,
for tests / dry runs).
"""

from lumintelligence.llm.base import (
    EmbeddingProvider,
    EmbedPurpose,
    LLMProvider,
    json_array_schema,
    json_object_schema,
)
from lumintelligence.llm.mock import MockProvider
from lumintelligence.llm.openai import OpenAIProvider
from lumintelligence.llm.registry import available_providers, make_provider, register_provider

__all__ = [
    "EmbedPurpose",
    "EmbeddingProvider",
    "LLMProvider",
    "MockProvider",
    "OpenAIProvider",
    "available_providers",
    "json_array_schema",
    "json_object_schema",
    "make_provider",
    "register_provider",
]
