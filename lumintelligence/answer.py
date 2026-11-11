"""Answer synthesis: a cited answer from the query's source excerpts only.

The chat model receives the question and numbered excerpts (bundled prompt ``sp_Answer``) and
returns ``{"answer", "cited"}``. Citation numbers that don't exist are dropped from ``cited``.
"""

from __future__ import annotations

from lumintelligence.llm.base import LLMProvider

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "cited": {"type": "array", "items": {"type": "integer"}}},
    "required": ["answer", "cited"],
    "additionalProperties": False,
}

NO_SOURCES = "The knowledge base has nothing relevant to this question."


def build_context(question: str, excerpts: list[dict]) -> str:
    """``excerpts``: ``{"n", "content", "label"?, "where"?}``."""
    parts = [f"Question: {question}", "", "Excerpts:"]
    for e in excerpts:
        head = f"[{e['n']}]"
        if e.get("label"):
            head += f" ({e['label']}{', ' + e['where'] if e.get('where') else ''})"
        parts += [head, e["content"].strip(), ""]
    return "\n".join(parts)


def synthesize(llm: LLMProvider, prompt: str, question: str, excerpts: list[dict]) -> dict:
    """``{"answer", "cited"}``; a fixed reply without a model call when there are no excerpts."""
    if not excerpts:
        return {"answer": NO_SOURCES, "cited": []}
    result = llm.complete_json(prompt, build_context(question, excerpts), SCHEMA)
    if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
        raise TypeError(f"answer model returned an unexpected result: {result!r}"[:300])
    valid = {e["n"] for e in excerpts}
    cited = [n for n in result.get("cited") or [] if isinstance(n, int) and n in valid]
    return {"answer": result["answer"].strip(), "cited": cited}
