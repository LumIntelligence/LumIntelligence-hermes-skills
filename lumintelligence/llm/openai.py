"""OpenAIProvider — LLM completions + embeddings via the openai SDK, against OpenAI or any
OpenAI-compatible endpoint (``OpenAIConfig.base_url``: Ollama, LM Studio, llama.cpp, vLLM, MLX).

LLM uses JSON mode (response_format={"type": "json_object"}) for structured output.
Embeddings use text-embedding-3-small (fixed per provider; no task-type mapping needed).
Token usage is tracked and reported via the optional on_tokens callback.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

from lumintelligence.config import OpenAIConfig
from lumintelligence.llm._retry import with_retry
from lumintelligence.llm.base import EmbeddingProvider, EmbedPurpose, LLMProvider, OutputTruncated


class OpenAIProvider(LLMProvider, EmbeddingProvider):
    def __init__(
        self,
        api_key: str | None = None,
        config: OpenAIConfig | None = None,
        embed_batch_size: int = 128,
        on_tokens: Callable[[str, int, int], None] | None = None,
    ):
        self.config = config or OpenAIConfig()
        key = api_key or os.environ.get(self.config.api_key_env)
        if not key and self.config.base_url:
            key = "not-needed"  # local OpenAI-compatible servers ignore the key, but the SDK requires one
        if not key:
            raise ValueError(
                f"No OpenAI API key — pass api_key=... or set ${self.config.api_key_env}."
            )
        from openai import OpenAI
        # Disable SDK-level retries — our with_retry handles the full policy
        # (honors Retry-After, classifies error types, applies correct back-off).
        # Leaving SDK retries on multiplies attempts uncontrollably.
        # Explicit timeout (found missing 2026-08-14, live) — the SDK's own default (10
        # minutes) meant a single stuck call could block an ingest job's single background
        # thread for a very long time with nothing raised for with_retry/the job's own
        # except-Exception boundary to catch, leaving the job stuck at "running" rather than
        # cleanly failing. 60s is generous for a single completion/embedding call; with_retry
        # still gets to apply its own backoff/retry policy on top of each bounded attempt.
        self._client = OpenAI(api_key=key, base_url=self.config.base_url, max_retries=0,
                              timeout=self.config.timeout_s)
        self._embed_batch_size = embed_batch_size
        # `kind` ("llm" | "embedding") is passed alongside token counts so a caller can
        # attribute usage without guessing from completion_tokens being 0 (embedding calls
        # always report 0 there).
        self._on_tokens = on_tokens or (lambda kind, i, o: None)
        self._schema_rejected = False   # set once the server refuses json_schema; then json_object

    # ------------------------------------------------------------------- LLM
    def complete_json(self, system_prompt: str, content: str, schema: dict) -> dict | list:
        # OpenAI JSON mode can only return a top-level object, never a bare array.
        # For array schemas, append a wrapper instruction and unwrap the result.
        is_array = schema.get("type") == "array"
        if is_array:
            system_prompt = system_prompt + '\nReturn a JSON object with a single key "items" whose value is the array.'
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
        strict = self._strict_format(schema)
        try:
            resp = self._create_json(messages, strict or {"type": "json_object"})
        except Exception as exc:
            if strict is None or not _is_bad_request(exc):
                raise
            # The server does not accept json_schema (or this schema): JSON mode from now on;
            # callers validate the shape themselves.
            self._schema_rejected = True
            resp = self._create_json(messages, {"type": "json_object"})
        self._pace()
        if resp.usage:
            self._on_tokens("llm", resp.usage.prompt_tokens or 0, resp.usage.completion_tokens or 0)
        if resp.choices[0].finish_reason == "length":
            raise OutputTruncated("answer cut off at the server's length limit")
        result = json.loads(resp.choices[0].message.content or "null")
        if is_array and isinstance(result, dict):
            # Unwrap {"items": [...]} or any single-key dict whose value is a list.
            if "items" in result and isinstance(result["items"], list):
                return result["items"]
            for v in result.values():
                if isinstance(v, list):
                    return v
        return result

    def _create_json(self, messages: list[Any], response_format: Any):
        return with_retry(lambda: self._client.chat.completions.create(
            model=self.config.llm_model, messages=messages, response_format=response_format,
            **self._reasoning_kwargs(),
        ))

    def _strict_format(self, schema: dict) -> dict | None:
        """Strict json_schema response format for closed object schemas (every property listed
        and required, e.g. batched tagging); None for open shapes (NoD's dynamic keys, arrays)."""
        if (not self.config.structured_output or self._schema_rejected or schema.get("type") != "object"
                or "properties" not in schema or schema.get("additionalProperties") is not False):
            return None
        return {"type": "json_schema", "json_schema": {"name": "result", "strict": True, "schema": schema}}

    def complete(self, system_prompt: str, content: str) -> str:
        resp = with_retry(lambda: self._client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            **self._reasoning_kwargs(),
        ))
        self._pace()
        if resp.usage:
            self._on_tokens("llm", resp.usage.prompt_tokens or 0, resp.usage.completion_tokens or 0)
        return resp.choices[0].message.content or ""

    def _reasoning_kwargs(self) -> dict:
        effort = self.config.reasoning_effort
        return {"reasoning_effort": effort} if effort is not None else {}

    def _pace(self) -> None:
        if self.config.request_pause_s:
            time.sleep(self.config.request_pause_s)

    # ------------------------------------------------------------- embeddings
    @property
    def dimension(self) -> int:
        return self.config.embedding_dim

    @property
    def model_id(self) -> str | None:
        return self.config.embedding_model

    def embed(self, texts: list[str], purpose: EmbedPurpose) -> list[list[float]]:
        if not texts:
            return []
        prefix = _task_prefix(self.config.embedding_model, purpose)
        if prefix:
            texts = [prefix + t for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(texts), self._embed_batch_size):
            chunk = texts[i : i + self._embed_batch_size]
            out.extend(self._embed_chunk(chunk))
            self._pace()
        return out

    def _embed_chunk(self, chunk: list[str]) -> list[list[float]]:
        # text-embedding-3-* support the Matryoshka `dimensions` param. Pass the configured
        # embedding_dim so vectors match the vector store's dimension — critical when a gemini
        # dataset (768-dim index) is transparently served by OpenAI: without this the SDK
        # returns native 1536-dim vectors and `embedding match` fails on the 768-dim table.
        # ada-002 doesn't accept the param, so only send it for the -3 family.
        kwargs: dict = {"model": self.config.embedding_model, "input": chunk}
        if self.config.embedding_model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.config.embedding_dim
        resp = with_retry(lambda: self._client.embeddings.create(**kwargs))
        if resp.usage:
            self._on_tokens("embedding", resp.usage.total_tokens or 0, 0)
        return [item.embedding for item in resp.data]


def _task_prefix(model: str, purpose: EmbedPurpose) -> str:
    """Task prefixes some open embedding models expect (nomic-embed: search_document / search_query /
    clustering); empty for models that don't use them."""
    if "nomic-embed" not in model:
        return ""
    return {EmbedPurpose.QUERY: "search_query: ", EmbedPurpose.CLUSTERING: "clustering: "}.get(
        purpose, "search_document: ")


def _is_bad_request(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 400 or "400" in str(exc)[:40]
