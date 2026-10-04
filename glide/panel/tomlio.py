"""A small TOML writer for the panel: the data model is what `tomllib` reads (tables, lists, strings, numbers, booleans).

Comments in a file are not kept (the panel says so before it writes, and shows the diff). The output is checked by
reading it back: if it does not read back to the same data, nothing is written.
"""

from __future__ import annotations

import difflib
import json
import re
import tomllib
from collections.abc import Mapping

_BARE = re.compile(r"[A-Za-z0-9_-]+")


class TomlError(ValueError):
    pass


def _key(name: str) -> str:
    return name if _BARE.fullmatch(name) else json.dumps(name)


def _value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # JSON string escapes are TOML basic-string escapes
    if isinstance(value, list):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    if isinstance(value, Mapping):
        return "{ " + ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items()) + " }" if value else "{}"
    raise TomlError(f"cannot write a {type(value).__name__}")


def _table(lines: list[str], path: tuple[str, ...], data: Mapping) -> None:
    scalars = {k: v for k, v in data.items() if not isinstance(v, Mapping)}
    tables = {k: v for k, v in data.items() if isinstance(v, Mapping)}
    if path and (scalars or not tables):
        lines += [f"[{'.'.join(_key(p) for p in path)}]"]
    for k, v in scalars.items():
        if isinstance(v, list) and any(isinstance(x, Mapping) for x in v) and len(v) > 1:
            lines += [f"{_key(k)} = [", *[f"  {_value(x)}," for x in v], "]"]
        else:
            lines.append(f"{_key(k)} = {_value(v)}")
    if scalars or (path and not tables):
        lines.append("")
    for k, v in tables.items():
        _table(lines, (*path, k), v)


def dumps(data: Mapping, header: str = "") -> str:
    lines: list[str] = [header, ""] if header else []
    _table(lines, (), data)
    text = "\n".join(lines).rstrip() + "\n"
    try:
        back = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise TomlError(f"the generated file is not valid TOML: {error}") from None
    if back != json.loads(json.dumps(data)):
        raise TomlError("the generated file does not read back to the same settings")
    return text


def diff(old: str, new: str) -> str:
    out = difflib.unified_diff(old.splitlines(), new.splitlines(), "glide.toml (now)", "glide.toml (after)", lineterm="", n=2)
    return "\n".join(out)
