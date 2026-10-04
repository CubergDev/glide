"""The shapes of the file planner: an immutable plan, the approval that unlocks exactly that plan, and the report."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .safety import Refused

Stamp = tuple[int, int, int]  # inode, size, mtime in ns: what the file looked like when it was planned


@dataclass(frozen=True)
class Move:
    source: str  # absolute path of a direct child of the plan's root
    destination: str  # absolute path: <root>/<folder>/<name>
    stamp: Stamp


@dataclass(frozen=True)
class Plan:
    root: str  # the real path of the folder being organized
    intent: str
    moves: tuple[Move, ...]
    plan_hash: str

    def compute_hash(self) -> str:
        """A digest of everything the plan will do. 32 hex characters: short enough to type, 128 bits."""
        body = {"root": self.root, "intent": self.intent, "moves": [[m.source, m.destination, list(m.stamp)] for m in self.moves]}
        return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:32]

    def to_json(self) -> str:
        data = {"root": self.root, "intent": self.intent, "plan_hash": self.plan_hash}
        data["moves"] = [{"source": m.source, "destination": m.destination, "stamp": list(m.stamp)} for m in self.moves]
        return json.dumps(data, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Plan:
        """Read a plan back. A plan whose content no longer matches its hash is refused."""
        try:
            data = json.loads(text)
            moves = tuple(Move(m["source"], m["destination"], tuple(m["stamp"])) for m in data["moves"])
            plan = cls(data["root"], data["intent"], moves, data["plan_hash"])
            valid = all(isinstance(v, str) for v in (plan.root, plan.intent, plan.plan_hash)) and all(
                isinstance(v, str) for m in moves for v in (m.source, m.destination)
            )
            stamps_ok = all(len(m.stamp) == 3 and all(isinstance(n, int) for n in m.stamp) for m in moves)
        except (ValueError, KeyError, TypeError) as error:
            raise Refused("that is not a readable plan") from error
        if not (valid and stamps_ok) or plan.compute_hash() != plan.plan_hash:
            raise Refused("the plan does not match its hash: it was changed after it was made")
        return plan


@dataclass(frozen=True)
class Approval:
    """A person's yes to exactly one plan. `execute` runs nothing without one that carries the plan's hash."""

    plan_hash: str


@dataclass(frozen=True)
class ActionResult:
    source: str
    destination: str
    status: str  # done | failed | uncertain | skipped
    reason: str = ""  # short and free of file content; empty for done


@dataclass(frozen=True)
class Report:
    status: str  # ok | partial | failed
    actions: tuple[ActionResult, ...]
    manifest: Path | None = None  # the undo manifest, when anything was attempted
    folders_created: tuple[str, ...] = ()
    folders_removed: tuple[str, ...] = ()
