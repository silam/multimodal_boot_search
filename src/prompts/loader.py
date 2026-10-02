"""Load versioned prompt templates by name. Format: system part, a `---` line, then user part."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPTS_DIR = Path(__file__).parent
_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR), undefined=StrictUndefined, keep_trailing_newline=True
)


@lru_cache
def _split(name: str) -> tuple[str, str]:
    raw = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    system, _, user = raw.partition("\n---\n")
    return system, user


def render(name: str, **variables: Any) -> tuple[str, str]:
    """Return (system, user) messages for prompt `name` rendered with `variables`."""
    system_tpl, user_tpl = _split(name)
    return (
        _env.from_string(system_tpl).render(**variables).strip(),
        _env.from_string(user_tpl).render(**variables).strip(),
    )
