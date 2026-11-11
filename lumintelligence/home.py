"""The LumIntelligence home folder: settings in ``.env``, the one KB, and its side files.

``~/.lumintelligence/`` (or ``LUMINTELLIGENCE_HOME``) holds ``.env``, ``kb.db``, ``status.json``,
``queue.json``, ``journal.jsonl``, ``worker.log`` and ``.lock``. Settings come from built-in
defaults, then ``.env``, then real environment variables (highest).
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import urllib.request
from pathlib import Path

from lumintelligence.config import Config, OpenAIConfig

HOME_ENV = "LUMINTELLIGENCE_HOME"

MODES = ("offline", "online")
MODE_FIELDS = ("BASE_URL", "API_KEY", "CHAT_MODEL", "EMBED_MODEL", "REASONING_EFFORT")
OLLAMA_URL = "http://localhost:11434/v1"

MODE_NOTE = ("Switching mode also switches the embedding model. A knowledge base only works with the "
             "embedding model it was built with: after switching, run `lum kb rebuild` (deletes everything) "
             "and ingest your files again, or switch back to keep using the old knowledge base.")

# key -> (default, comment written above it in .env; "## ..." starts a section)
SETTINGS: dict[str, tuple[str, str]] = {
    "LUM_MODE": ("offline", "## Which set of models to use: offline (Ollama on this computer) or online "
                            "(a hosted API).\n" + MODE_NOTE),
    "LUM_OFFLINE_BASE_URL": (OLLAMA_URL, "## offline: Ollama"),
    "LUM_OFFLINE_API_KEY": ("", ""),
    "LUM_OFFLINE_CHAT_MODEL": ("hermes3", ""),
    "LUM_OFFLINE_EMBED_MODEL": ("nomic-embed-text", ""),
    "LUM_OFFLINE_REASONING_EFFORT": ("", ""),
    "LUM_ONLINE_BASE_URL": ("", "## online: empty base URL = OpenAI; any OpenAI-compatible provider works"),
    "LUM_ONLINE_API_KEY": ("", ""),
    "LUM_ONLINE_CHAT_MODEL": ("gpt-6-luna", ""),
    "LUM_ONLINE_EMBED_MODEL": ("text-embedding-3-small", ""),
    "LUM_ONLINE_REASONING_EFFORT": ("none", ""),
    "LUM_MAX_IN_FLIGHT": ("2", ("## build settings (both modes)\nmodel calls in parallel; match your "
                                "server's parallel slots")),
    "LUM_PAGE_FILTER": ("true", "PDF: skip contents, cover and blank pages using the chat model"),
    "LUM_CHUNK_WINDOW_LINES": ("15", "lines per source chunk"),
    "LUM_CHUNK_OVERLAP_LINES": ("3", "lines shared by consecutive chunks (at most a fifth of the size)"),
    "LUM_CHUNK_MAX_CHARS": ("1500", "characters per source chunk; longer paragraphs are split at sentences"),
    "LUM_CONFIRM_ABOVE": ("20", "ingest asks to confirm when more new/changed files than this"),
    "LUM_PROMPTS_DIR": ("", ("## custom prompts (guide: ~/.hermes/skills/research/lumintelligence/references/custom_prompts.md)\n"
                             "Empty = the default prompts that come with the skill. Or a folder of your own "
                             "prompt files (sp_NoD.txt, sp_ADHD.txt, sp_PageFilter.txt, sp_Intention_Engine.txt, "
                             "sp_Answer.txt); each replaces the default of the same name, missing files keep "
                             "the default. Extraction, tagging and page-filter prompts apply to files added "
                             "afterwards; question and answer prompts apply immediately. `lum prompts export "
                             "<folder>` copies the defaults as a starting point.")),
}
PROVIDER_KEY = "LUM_PROVIDER"   # "openai" (default) or "mock" (offline tests); not written to .env
ENTRY_CUTOFF_KEY = "LUM_ENTRY_CUTOFF"   # optional override of the query entry cutoff; not written to .env


class SetupError(RuntimeError):
    """A user-fixable problem; ``hint`` says what to do."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


def home() -> Path:
    return Path(os.environ.get(HOME_ENV) or "~/.lumintelligence").expanduser()


def path(name: str) -> Path:
    """A file in the home folder: ``.env``, ``kb.db``, ``status.json``, ``queue.json``, ..."""
    return home() / name


# ------------------------------------------------------------------------- .env
_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def read_env(file: Path | None = None) -> dict[str, str]:
    file = file or path(".env")
    out: dict[str, str] = {}
    if not file.exists():
        return out
    for raw in file.read_text(encoding="utf-8").splitlines():
        if raw.lstrip().startswith("#"):
            continue
        m = _LINE.match(raw)
        if not m:
            continue
        value = m.group(2)
        if value[:1] in ("'", '"') and value[-1:] == value[:1]:
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        out[m.group(1)] = value
    return out


def write_env(values: dict[str, str], file: Path | None = None) -> None:
    """Write ``.env`` with every known setting and its comment (chmod 600). Unknown keys kept."""
    file = file or path(".env")
    file.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# LumIntelligence settings. Real environment variables override these values."]
    for key, (default, comment) in SETTINGS.items():
        for i, c in enumerate(comment.split("\n") if comment else []):
            if c.startswith("## "):
                lines += ["", f"# --- {c[3:]} ---"]
            else:
                lines += textwrap.wrap(c, 98, initial_indent="# ", subsequent_indent="# ") if i or c else []
        lines.append(f"{key}={values.get(key, default)}")
    extra = {k: v for k, v in values.items() if k not in SETTINGS}
    if extra:
        lines.append("")
        lines += [f"{k}={v}" for k, v in extra.items()]
    tmp = file.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, file)


def set_env_value(key: str, value: str, file: Path | None = None) -> None:
    """Change one ``KEY=`` line, keeping every other line and comment as is."""
    file = file or path(".env")
    lines = file.read_text(encoding="utf-8").splitlines() if file.exists() else []
    for i, raw in enumerate(lines):
        m = _LINE.match(raw)
        if m and m.group(1) == key and not raw.lstrip().startswith("#"):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(file, 0o600)


def settings() -> dict[str, str]:
    s = {k: d for k, (d, _) in SETTINGS.items()}
    s.update(read_env())
    s.update({k: v for k, v in os.environ.items() if k.startswith("LUM_")})
    return s


def initialized() -> bool:
    return path(".env").exists()


def require_init() -> dict[str, str]:
    if not initialized():
        raise SetupError(f"LumIntelligence is not set up ({path('.env')} not found).", "Run `lum init`.")
    return settings()


# -------------------------------------------------------------------- providers
def _bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(s: dict[str, str], key: str) -> int:
    try:
        return int(s[key])
    except (KeyError, ValueError) as exc:
        raise SetupError(f"{key} must be a whole number (got {s.get(key)!r}).", f"Fix it in {path('.env')}.") from exc


def int_setting(s: dict[str, str], key: str) -> int:
    return _int(s, key)


def mode(s: dict[str, str]) -> str:
    m = s.get("LUM_MODE", "offline").strip().lower()
    if m not in MODES:
        raise SetupError(f"LUM_MODE must be offline or online (got {m!r}).", f"Fix it in {path('.env')}.")
    return m


def resolve_models(s: dict[str, str]) -> dict[str, str]:
    """The active mode's set: ``mode``, ``base_url``, ``api_key``, ``chat_model``, ``embed_model``,
    ``reasoning_effort``."""
    m = mode(s)
    out = {"mode": m}
    for f in MODE_FIELDS:
        out[f.lower()] = s.get(f"LUM_{m.upper()}_{f}", "")
    return out


def make_config(s: dict[str, str], embedding_dim: int | None = None) -> Config:
    r = resolve_models(s)
    chat = OpenAIConfig(
        llm_model=r["chat_model"],
        reasoning_effort=r["reasoning_effort"] or None,
        embedding_model=r["embed_model"],
        embedding_dim=embedding_dim or OpenAIConfig().embedding_dim,
        base_url=r["base_url"] or None,
    )
    extra: dict = {"prompts_dir": prompts_dir(s)}
    if s.get(ENTRY_CUTOFF_KEY):
        extra["entry_cutoff"] = float(s[ENTRY_CUTOFF_KEY])
    return Config(openai=chat, max_in_flight=_int(s, "LUM_MAX_IN_FLIGHT"), **extra,
                  page_filter=_bool(s["LUM_PAGE_FILTER"]),
                  chunk_window_lines=_int(s, "LUM_CHUNK_WINDOW_LINES"),
                  chunk_overlap_lines=_int(s, "LUM_CHUNK_OVERLAP_LINES"),
                  chunk_max_chars=_int(s, "LUM_CHUNK_MAX_CHARS"))


def prompts_dir(s: dict[str, str]) -> str | None:
    """The custom prompts folder, or None for the defaults. A set but missing folder is an error."""
    raw = (s.get("LUM_PROMPTS_DIR") or "").strip()
    if not raw:
        return None
    folder = os.path.abspath(os.path.expanduser(raw))
    if not os.path.isdir(folder):
        raise SetupError(f"LUM_PROMPTS_DIR points to {folder}, which is not a folder.",
                         f"Create it (e.g. `lum prompts export {raw}`), or empty LUM_PROMPTS_DIR in "
                         f"{path('.env')} to use the default prompts.")
    return folder


def kb_meta(key: str) -> str | None:
    """A value from ``kb.db``'s meta table, or None (no KB yet)."""
    if not path("kb.db").exists():
        return None
    from lumintelligence.storage.sqlite_conn import connect

    conn = connect(str(path("kb.db")))
    try:
        row = conn.execute("select value from meta where key = ?", (key,)).fetchone()
    except Exception:  # noqa: BLE001 — no meta table yet
        row = None
    finally:
        conn.close()
    return row[0] if row else None


def make_providers(s: dict[str, str], embedding_dim: int | None = None):
    """``(chat, embedder)`` for the active mode. The embedding size comes from ``kb.db`` when the KB
    exists, else from one probe call (``text-embedding-3-*``: the configured default)."""
    if s.get(PROVIDER_KEY, "openai") == "mock":
        from lumintelligence.llm.registry import make_provider
        p = make_provider("mock")
        return p, p
    from lumintelligence.llm.openai import OpenAIProvider

    r = resolve_models(s)
    if embedding_dim is None and r["embed_model"] == kb_meta("embed_model"):
        stored = kb_meta("embedding_dim")
        embedding_dim = int(stored) if stored else None
    cfg = make_config(s, embedding_dim)
    try:
        chat = OpenAIProvider(api_key=r["api_key"] or None, config=cfg.openai)
    except ValueError as exc:
        raise SetupError(str(exc), f"Set LUM_{r['mode'].upper()}_API_KEY in {path('.env')}.") from exc
    if embedding_dim is None and not r["embed_model"].startswith("text-embedding-3"):
        from lumintelligence.llm.base import EmbedPurpose
        try:
            cfg.openai.embedding_dim = len(chat.embed_one("dimension probe", EmbedPurpose.QUERY))
        except Exception as exc:
            raise SetupError(f"The embedding model {r['embed_model']!r} did not answer: {exc}"[:400],
                             "Check that the model is installed and the server is running, then `lum check`.") from exc
    return chat, chat


def ollama_status(base_url: str, timeout: float = 3.0) -> dict:
    """``{"reachable": bool, "models": [names]}`` from Ollama's ``/api/tags`` (base URL may end in /v1)."""
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=timeout) as resp:
            data = json.load(resp)
    except Exception:  # noqa: BLE001 — any failure means "not reachable"
        return {"reachable": False, "models": []}
    return {"reachable": True, "models": [m.get("name", "") for m in data.get("models", [])]}


def missing_models(installed: list[str], wanted: list[str]) -> list[str]:
    """Wanted names not installed; "hermes3" matches "hermes3:latest", a tag must match exactly."""
    have = set(installed) | {n.split(":")[0] for n in installed if n.endswith(":latest")}
    return [w for w in wanted if w not in have]


# --------------------------------------------------------------------- the KB
def open_kb(s: dict[str, str] | None = None):
    """Open (or create) the one KB with the configured models. Refuses an embedding model other than
    the one the KB was built with."""
    from lumintelligence.facade import LumIntelligence

    s = s or require_init()
    r = resolve_models(s)
    built_with = kb_meta("embed_model")
    if built_with is not None and built_with != r["embed_model"]:
        built_mode = kb_meta("embed_mode") or "another mode"
        raise SetupError(
            f"This knowledge base was built with embedding model {built_with!r} ({built_mode}); the active "
            f"mode ({r['mode']}) uses {r['embed_model']!r}.",
            "Switch LUM_MODE back, or run `lum kb rebuild` in a terminal to start over (deletes everything).")
    chat, embed = make_providers(s)
    cfg = make_config(s, embed.dimension)
    try:
        lum = LumIntelligence.open(str(home()), chat, embed, config=cfg)
    except ValueError as exc:
        raise SetupError(str(exc), "Use the embedding model the KB was built with, or start over with "
                                   "`lum kb rebuild`.") from exc
    conn = lum.graph.conn  # type: ignore[attr-defined]
    for key, value in (("embed_model", r["embed_model"]), ("embed_mode", r["mode"])):
        # read first: an insert, even a no-op one, waits for a running build's write lock
        if conn.execute("select 1 from meta where key = ?", (key,)).fetchone() is None:
            conn.execute("insert or ignore into meta(key, value) values (?, ?)", (key, value))
    return lum
