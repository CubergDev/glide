"""The few fixed sentences Glide says itself, in English, Cantonese and Mandarin.

Everything else it says is written by a model in the user's language. These are the sentences that report
a state of Glide (a dry run, a task that stalled, a provider that is down), which no model is asked for.
Cantonese is written the way it is spoken in Hong Kong, in traditional characters; Mandarin in simplified.
A language with no entry gets English.
"""

# ruff: noqa: RUF001  full-width punctuation is the right punctuation here

from __future__ import annotations

PHRASES: dict[str, dict[str, str]] = {
    "dry_run": {
        "en": "Dry run, nothing was done. The first move would be: {what}.",
        "yue": "試行模式，未有做任何嘢。第一步會係：{what}。",
        "zh": "试运行，什么都没有做。第一步会是：{what}。",
    },
    "dry_run_plain": {
        "en": "Dry run, nothing was done.",
        "yue": "試行模式，未有做任何嘢。",
        "zh": "试运行，什么都没有做。",
    },
    "done": {"en": "Done.", "yue": "搞掂喇。", "zh": "完成了。"},
    "nothing": {
        "en": "I could not find anything on screen that helps.",
        "yue": "喺螢幕上搵唔到有用嘅嘢。",
        "zh": "屏幕上没有找到有用的东西。",
    },
    "unsure": {
        "en": "I was not sure what to do next, so I stopped.",
        "yue": "我唔肯定下一步點做，所以停低咗。",
        "zh": "我不确定下一步该怎么做，所以停下了。",
    },
    "stalled": {
        "en": "I got stuck and stopped.",
        "yue": "我卡住咗，所以停低咗。",
        "zh": "我卡住了，所以停下了。",
    },
    "limit": {
        "en": "I ran out of steps before finishing.",
        "yue": "未做完就已經用晒步數上限。",
        "zh": "步数用完了，还没有做完。",
    },
    "provider": {
        "en": "I could not finish because a service failed.",
        "yue": "有服務出錯，所以做唔完。",
        "zh": "有服务出错，没能完成。",
    },
    "crashed": {
        "en": "Something went wrong, so I stopped.",
        "yue": "出咗啲問題，所以停低咗。",
        "zh": "出了点问题，所以停下了。",
    },
    "not_configured": {
        "en": "A provider is not set up, so I cannot do that yet.",
        "yue": "有啲服務未設定好，所以暫時做唔到。",
        "zh": "有服务还没有设置好，暂时做不了。",
    },
    "no_permission": {
        "en": "I need Accessibility permission to act on this Mac.",
        "yue": "我需要輔助使用權限先可以操作呢部 Mac。",
        "zh": "我需要辅助功能权限才能操作这台 Mac。",
    },
    "busy": {
        "en": "I am already working on a task. Say stop first.",
        "yue": "我已經喺度做緊一件事，請先講停。",
        "zh": "我正在做另一件事，请先说停。",
    },
    "no_llm": {
        "en": "Sorry, I cannot reach my language models right now.",
        "yue": "對唔住，我而家連唔到語言模型。",
        "zh": "抱歉，我现在连不上语言模型。",
    },
    "no_speech": {
        "en": "Sorry, I did not catch that.",
        "yue": "對唔住，我聽唔清楚。",
        "zh": "抱歉，我没听清。",
    },
    "stopped": {"en": "Stopped.", "yue": "停咗。", "zh": "已停止。"},
}


def say(key: str, language: str | None = None, **fields: str) -> str:
    """The sentence for `key` in `language` ('yue', 'zh-HK', 'zh', ...), English when there is none."""
    table = PHRASES[key]
    code = (language or "en").lower().replace("_", "-")
    text = table.get(code) or table.get(code.split("-")[0]) or table["en"]
    return text.format(**fields) if fields else text
