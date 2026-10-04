"""The few sentences the router's callers say when a request cannot go ahead: English, Cantonese and Mandarin.

Same rules as `glide/assistant/phrases.py`: Cantonese is written the way it is spoken in Hong Kong, in traditional
characters, Mandarin in simplified, and a language with no entry gets English. These are moved into that file's table when the
integration plan is applied (docs/ROUTER.md); they are kept apart here so this package imports nothing from
`glide.assistant`.
"""

# ruff: noqa: RUF001  full-width punctuation is the right punctuation here

from __future__ import annotations

PHRASES: dict[str, dict[str, str]] = {
    "clarify_default": {
        "en": "I am not sure what you want me to do. Could you say a little more?",
        "yue": "我唔肯定你想我做啲咩，可唔可以講多少少？",
        "zh": "我不确定你想让我做什么，能再说详细一点吗？",
    },
    "clarify_needed": {
        "en": "I did not do anything, because I need to know this first: {question}",
        "yue": "我乜都未做，因為我要先知道：{question}",
        "zh": "我什么都没有做，因为我需要先知道：{question}",
    },
}

LANGUAGES = {"yue": "yue", "zh": "zh", "cmn": "zh", "en": "en"}


def say(key: str, language: str | None = None, **values: str) -> str:
    """A fixed sentence in `language` (English when there is none), with `values` filled in."""
    table = PHRASES[key]
    code = (language or "en").lower().split("-")[0]
    return table.get(LANGUAGES.get(code, "en"), table["en"]).format(**values)
