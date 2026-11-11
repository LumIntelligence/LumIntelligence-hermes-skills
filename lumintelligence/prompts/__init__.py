"""Bundled prompt files, a loader, and custom-prompt resolution.

``load_prompt("sp_NoD")`` reads a bundled prompt as text. Names may omit the ``.txt``.
Stage-owned prompts (NoD, ADHD) live alongside their stage module
(``lumintelligence.nod.prompts``, ``lumintelligence.adhd.prompts``); when no explicit
``package`` is given, those are searched too.

``resolve_prompt(value, prompts_dir)`` is the custom-prompt intake: an explicit file path,
else a file in ``prompts_dir``, else a bundled prompt. Per-call overrides (full prompt text)
bypass this entirely; see the facade. ``fill_placeholders`` substitutes ``{{ATTENTION_HINTS}}``
and ``{{DATASET_DESCRIPTION}}``.
"""

import re
from importlib.resources import files
from pathlib import Path

_DEFAULT_SEARCH_PACKAGES: tuple[str, ...] = (
    "lumintelligence.prompts",
    "lumintelligence.nod.prompts",
    "lumintelligence.adhd.prompts",
)


def load_prompt(name: str, package: str | None = None) -> str:
    if not name.endswith(".txt"):
        name += ".txt"
    packages = (package,) if package else _DEFAULT_SEARCH_PACKAGES
    for pkg in packages:
        resource = files(pkg) / name
        if resource.is_file():
            return resource.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"prompt {name!r} not found in {', '.join(packages)}")


def resolve_prompt(value: str, prompts_dir: str | None = None) -> str:
    """Resolve a prompt setting: explicit file path -> file in prompts_dir -> bundled prompt."""
    if "/" in value or value.endswith(".txt"):
        p = Path(value).expanduser()
        if p.is_file():
            return p.read_text(encoding="utf-8")
    if prompts_dir:
        candidate = Path(prompts_dir).expanduser() / (value if value.endswith(".txt") else f"{value}.txt")
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return load_prompt(Path(value).name if value.endswith(".txt") else value)


def fill_placeholders(prompt: str, *, attention_hints: str | None = None,
                      dataset_description: str | None = None) -> str:
    if attention_hints is not None:
        prompt = prompt.replace("{{ATTENTION_HINTS}}", attention_hints)
    if dataset_description is not None:
        prompt = prompt.replace("{{DATASET_DESCRIPTION}}", dataset_description)
    return prompt


# Prompts a user may replace through ``prompts_dir`` (LUM_PROMPTS_DIR). Ingest-time ones shape what is
# stored; question-time ones apply to every query immediately. ``sp_ADHD_frame`` is engine-owned.
INGEST_PROMPTS = ("sp_NoD", "sp_ADHD", "sp_PageFilter")
QUERY_PROMPTS = ("sp_Intention_Engine", "sp_Answer")
CUSTOMIZABLE = INGEST_PROMPTS + QUERY_PROMPTS
ENGINE_OWNED = ("sp_ADHD_frame",)


def prompt_source(name: str, prompts_dir: str | None) -> str:
    """``"custom"`` if ``prompts_dir`` holds ``<name>.txt``, else ``"default"``."""
    if prompts_dir and (Path(prompts_dir).expanduser() / f"{name}.txt").is_file():
        return "custom"
    return "default"


_HEADING = re.compile(r"^\s*(\*\*[^*]+:\*\*|#{1,6}\s+\S)")
_OUTPUT_HEADING = re.compile(r"^\s*(\*\*\s*output format\s*:\s*\*\*|#{1,6}\s+output format\b)", re.IGNORECASE)


def output_section(text: str) -> str | None:
    """The "Output format" section (its heading line up to the next heading), whitespace-normalized;
    None if the prompt has none."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _OUTPUT_HEADING.match(line):
            end = next((j for j in range(i + 1, len(lines)) if _HEADING.match(lines[j])), len(lines))
            return " ".join(" ".join(lines[i:end]).split())
    return None


__all__ = ["CUSTOMIZABLE", "ENGINE_OWNED", "INGEST_PROMPTS", "QUERY_PROMPTS", "fill_placeholders",
           "load_prompt", "output_section", "prompt_source", "resolve_prompt"]
