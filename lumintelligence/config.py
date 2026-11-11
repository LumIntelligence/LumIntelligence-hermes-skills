"""Central configuration — the *tunable* knobs.

Everything here is a value you might change per run or per project: models, endpoint,
embedding dimension, thresholds, prompts, including the "auto" thresholds that depend on the
embedding model.

Providers: one OpenAI-compatible interface. OpenAI itself is the default; local runtimes
(Ollama, LM Studio, llama.cpp server, vLLM, MLX servers) are reached by setting ``base_url``.

Distinct from :mod:`lumintelligence.constants`, which holds the *fixed* schema vocabulary
(node/edge types, attribute keys).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace


@dataclass
class OpenAIConfig:
    llm_model: str = "gpt-6-luna"
    # Sent as Chat Completions `reasoning_effort` when not None. GPT-6 models default to
    # "medium" when it's omitted; set None for local servers that reject the parameter.
    reasoning_effort: str | None = "none"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    api_key_env: str = "OPENAI_API_KEY"
    # None = OpenAI. Any OpenAI-compatible endpoint, e.g. "http://localhost:11434/v1" (Ollama).
    base_url: str | None = None
    # Send a strict json_schema when the caller's schema is closed (batched tagging); falls back
    # to JSON mode if the server rejects it.
    structured_output: bool = True
    request_pause_s: float = 0.0   # fixed pause after each call; 0 (the governor paces)
    max_retries: int = 3
    timeout_s: float = 60.0


@dataclass
class Config:
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)

    # Thresholds. "auto" resolves per embedder (see resolved()).
    consolidation_threshold: float | str = "auto"   # centroid nearest-fit cutoff (cosine similarity)
    entry_cutoff: float | str = "auto"              # query entry-point cutoff (cosine distance)
    n_per_keyword: int = 25                          # query entry points per keyword

    # Parallel build
    max_in_flight: int = 2       # provider calls in flight; match the server's parallel slots
    tag_batch_max: int = 50      # entities per tagging call (1 = one entity per call)
    tag_batch_min: int = 5       # smallest batch when spreading entities across slots

    # File input layer
    chunk_window_lines: int = 15     # lines per source chunk
    chunk_overlap_lines: int = 3     # lines shared by consecutive chunks
    chunk_max_chars: int = 1500      # characters per source chunk (long lines are split at sentences)
    page_filter: bool = True         # PDF: drop non-content pages (contents, cover, ...) via the chat model
    page_filter_prompt: str = "sp_PageFilter"
    max_file_mb: float | None = None  # files larger than this are reported and skipped

    # Prompts: a bundled prompt name (e.g. "sp_NoD"), a file path, or a name looked up in
    # prompts_dir first. Per-call overrides (full prompt text) are accepted by the facade.
    nod_prompt: str = "sp_NoD"
    adhd_prompt: str = "sp_ADHD"
    intention_prompt: str = "sp_Intention_Engine"
    answer_prompt: str = "sp_Answer"
    prompts_dir: str | None = None
    # Additive steering merged into the NoD/ADHD prompts via {{ATTENTION_HINTS}}.
    nod_attention_hints: str = ""
    adhd_attention_hints: str = ""
    dataset_description: str = "a collection of records"

    @property
    def embedding_dim(self) -> int:
        return self.openai.embedding_dim

    def api_key(self) -> str | None:
        return os.environ.get(self.openai.api_key_env)

    def resolved(self) -> Config:
        """Copy with "auto" thresholds replaced by embedder-appropriate defaults
        (OpenAI/Gemini embedders 0.88 / 0.30, others 0.80 / 0.55)."""
        model = self.openai.embedding_model.lower()
        known = "text-embedding" in model or "gemini" in model
        c = replace(self)
        if str(c.consolidation_threshold) == "auto":
            c.consolidation_threshold = 0.88 if known else 0.80
        if str(c.entry_cutoff) == "auto":
            c.entry_cutoff = 0.30 if known else 0.55
        return c


# Default instance; callers construct their own to override.
default_config = Config()
