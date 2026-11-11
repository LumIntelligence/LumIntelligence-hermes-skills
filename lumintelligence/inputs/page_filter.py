"""Page filter: the configured chat model decides which PDF pages are real content.

Pages
under 20 characters are blank without a call. A failed call keeps the page, so an error never drops
content. Calls run in parallel through the governor and are journaled, so a rerun after a crash
replays them.
"""

from __future__ import annotations

from lumintelligence.llm.base import LLMProvider
from lumintelligence.llm.governor import Governor
from lumintelligence.storage.journal import Journal, NullJournal, journal_key

MIN_CHARS = 20
MAX_CHARS = 4000
_ATTEMPTS = 2

SCHEMA = {
    "type": "object",
    "properties": {"has_content": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["has_content", "reason"],
    "additionalProperties": False,
}


class PageFilter:
    def __init__(self, llm: LLMProvider, prompt: str, governor: Governor | None = None,
                 journal: Journal | None = None):
        self.llm = llm
        self.prompt = prompt
        self.governor = governor or Governor(1)
        self.journal = journal if journal is not None else NullJournal()
        self._model_id = str(getattr(getattr(llm, "config", None), "llm_model", type(llm).__name__))

    def classify(self, pages: list[str]) -> list[tuple[bool, str]]:
        """``(has_content, reason)`` per page, in page order."""
        out = []
        for res in self.governor.map_ordered(self._one, pages):
            out.append((True, f"page filter error, page kept: {res}") if isinstance(res, Exception) else res)
        return out

    def _one(self, text: str) -> tuple[bool, str]:
        if not text or len(text.strip()) < MIN_CHARS:
            return False, "blank or near-empty page"
        page = text[:MAX_CHARS]
        key = journal_key("page_filter", self._model_id, self.prompt, page)
        cached = self.journal.get(key)
        if cached is not None:
            return bool(cached["has_content"]), str(cached.get("reason", ""))
        last = "no answer"
        for _ in range(_ATTEMPTS):
            try:
                result = self.llm.complete_json(self.prompt, page, SCHEMA)
            except Exception as exc:  # noqa: BLE001 — any failure: retry, then keep the page
                last = str(exc)
                continue
            if isinstance(result, dict) and isinstance(result.get("has_content"), bool):
                answer = {"has_content": result["has_content"], "reason": str(result.get("reason", ""))}
                self.journal.put(key, "page_filter", answer)
                return answer["has_content"], answer["reason"]
            last = f"unexpected answer: {result!r}"[:200]
        return True, f"page filter error, page kept: {last}"
