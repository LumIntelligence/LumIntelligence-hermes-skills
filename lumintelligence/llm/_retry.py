"""Shared retry logic for all online LLM providers.

Retry matrix (applied at the provider layer so all pipeline stages benefit automatically):
  429 / rate-limit  : honor server Retry-After if present; else exponential back-off (1→2→4→8→16s)
  5xx server error  : exponential back-off with 2s base, max 3 retries
  network timeout   : fixed 2s delay, max 3 retries
  auth error        : fast-fail immediately (wrong key won't fix itself on retry)
  malformed JSON    : handled at the pipeline layer (5 re-prompts per call)
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")


def _parse_retry_after(text: str) -> float | None:
    """Extract a suggested wait (seconds) from a 429 error message.

    Covers Gemini ('retry in 23.6s', 'retryDelay: 23s') and
    OpenAI ('try again in 20s', 'try again in 1ms').
    Millisecond values are converted to seconds.
    """
    for pat in (
        r"retry.after['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s?",
        r"retry in (\d+(?:\.\d+)?)s",
        r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s?",
        r"try again in (\d+(?:\.\d+)?)ms",   # OpenAI millisecond hint → convert
        r"try again in (\d+(?:\.\d+)?)s",     # OpenAI second hint
        r"please try again in (\d+(?:\.\d+)?)ms",
        r"please try again in (\d+(?:\.\d+)?)s",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if "ms" in pat:
                val = val / 1000.0
            return max(val, 0.1)   # floor at 100ms so we never sleep(0)
    return None


def _classify(exc: Exception) -> str:
    """Return 'auth', 'rate_limit', 'server', 'network', or 'other'."""
    msg = str(exc).lower()
    if any(x in msg for x in ("401", "403", "unauthorized", "forbidden", "invalid api key", "api_key")):
        return "auth"
    if any(x in msg for x in ("429", "rate_limit", "resource_exhausted", "too many requests", "quota")):
        return "rate_limit"
    if any(x in msg for x in ("500", "502", "503", "504", "internal server error", "bad gateway", "service unavailable")):
        return "server"
    if any(x in msg for x in ("timeout", "timed out", "connection", "network", "socket", "reset", "eof")):
        return "network"
    return "other"


def with_retry(
    call: Callable[[], _T],
    *,
    max_rate_limit_retries: int = 5,
    max_transient_retries: int = 3,
) -> _T:
    """Run ``call()`` with the standard retry policy. Re-raises on auth errors or exhaustion.

    Args:
        call: zero-argument callable wrapping the provider API call.
        max_rate_limit_retries: attempts for 429 / quota errors.
        max_transient_retries: attempts for 5xx / network errors.
    """
    rate_attempts = 0
    transient_attempts = 0

    while True:
        try:
            return call()
        except Exception as exc:
            kind = _classify(exc)

            if kind == "auth":
                raise  # wrong key; retrying won't help

            if kind == "rate_limit":
                rate_attempts += 1
                if rate_attempts >= max_rate_limit_retries:
                    raise
                suggested = _parse_retry_after(str(exc))
                wait = (suggested + 1.0) if suggested else min(60.0, 1.0 * (2 ** rate_attempts))
                time.sleep(wait)

            elif kind in ("server", "network"):
                transient_attempts += 1
                if transient_attempts >= max_transient_retries:
                    raise
                time.sleep(2.0 * transient_attempts)

            else:
                raise  # unknown error; propagate immediately
