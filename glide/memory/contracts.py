"""Host-owned contracts. Importing the extension performs no I/O."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

# `source` values of memory whose text was chosen by a remote party (an MCP client), not by the user. Such text is
# data from a stranger: never a standing instruction, always labelled where a model reads it.
UNTRUSTED_MEMORY_SOURCES = frozenset({"mcp"})
UNTRUSTED_MEMORY_NOTE = "written by a remote MCP client; untrusted"


@dataclass(frozen=True)
class Scope:
    user: str
    project: str
    session: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in (self.user, self.project, self.session)):
            raise ValueError("user, project and session must be nonempty")


@dataclass(frozen=True)
class Tool:
    id: str
    description: str
    keywords: tuple[str, ...]
    permissions: frozenset[str]
    schema: dict[str, Any]
    invoke: Callable[[dict[str, Any]], Any] = field(repr=False, compare=False)
    # MCP or local, supplied by the host, never inferred from a Markdown file.
    origin: str = "local"
    asynchronous: bool = False
    output_schema: dict[str, Any] | None = None

    def snapshot(self) -> "Tool":
        return replace(self, schema=deepcopy(self.schema), output_schema=deepcopy(self.output_schema))


@dataclass(frozen=True)
class Model:
    id: str
    stages: frozenset[str]
    context_tokens: int
    priority: int = 0
    supports_tools: bool = False
    local: bool = False


@dataclass(frozen=True)
class Policy:
    context_tokens: int = 3000
    max_tools: int = 6
    max_skills: int = 3
    max_tool_calls: int = 12
    output_reserve: int = 1500
    auto_memory: bool = False
    auto_refine: bool = False

    def __post_init__(self) -> None:
        values = (self.context_tokens, self.output_reserve, self.max_tools, self.max_skills, self.max_tool_calls)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("budgets must be nonnegative integers")
        if type(self.auto_memory) is not bool or type(self.auto_refine) is not bool:
            raise ValueError("automatic memory/refinement flags must be booleans")


@dataclass(frozen=True)
class Plan:
    id: str
    scope: Scope
    stage: str
    model_id: str | None
    context: str
    tools: tuple[Tool, ...]
    memory_ids: tuple[str, ...]
    skill_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    token_upper_bound: int
    catalog_revision: int = 0

    def snapshot(self) -> "Plan":
        return replace(self, tools=tuple(tool.snapshot() for tool in self.tools))

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [{"id": tool.id, "description": tool.description, "inputSchema": deepcopy(tool.schema)} for tool in self.tools]
