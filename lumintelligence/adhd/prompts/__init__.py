from lumintelligence.prompts import load_prompt as _load_prompt


def load_prompt(name: str = "sp_ADHD") -> str:
    return _load_prompt(name, package=__package__)


__all__ = ["load_prompt"]
