"""Calibration: what a tier's confidence is actually worth, from a table that was measured and not guessed.

A classifier says "execute, 0.9". How often is that right? Nobody knows until it has been run on labelled requests.
`Calibration` is the table of the answer: for each tier and predicted route, the raw confidence is cut into bins and
each bin keeps how many labelled cases landed in it and how many it got right. `apply` swaps a raw confidence for the
smoothed, monotone accuracy of its bin, so a thresholded decision means "right at least this often" and a
classifier that is overconfident (or underconfident) is corrected by what was measured.

- Smoothing: (right + 1) / (n + 2), so a bin of three correct cases is 0.8, not 1.0.
- Monotone: pool-adjacent-violators over the bins, so a higher raw confidence is never worth less than a lower one.
- Thin data is not trusted: a bin with fewer than `min_samples` cases falls back to the tier's pooled bin, and then to
  the raw number (reported as not calibrated). With no table at all, `Calibration.identity()` returns the raw number.

The table is data (JSON, `to_dict`/`from_dict`/`load`/`dump`). It is made by `fit` from `(tier, route, raw confidence,
was it right)` samples: offline from `tests/routing/eval.py` against fakes (which measures the fakes and nothing else),
and for real from a live run on the configured providers (docs/ROUTER.md). Nothing in this file knows a model.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EDGES = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DEFAULT_MIN_SAMPLES = 5
POOLED = "*"
VERSION = 1


def _key(tier: str, route: str) -> str:
    return f"{tier}:{route}"


@dataclass(frozen=True)
class Sample:
    tier: str
    route: str  # the route the tier predicted
    raw: float  # its raw confidence
    right: bool  # whether that route was the labelled one


def _bin(edges: Sequence[float], value: float) -> int:
    value = min(max(value, 0.0), 1.0)
    for i in range(len(edges) - 1):
        if value < edges[i + 1]:
            return i
    return len(edges) - 2  # 1.0 belongs to the last bin


def _monotone(counts: list[tuple[int, int]]) -> list[float | None]:
    """Accuracy per bin after pooling adjacent bins that break the order (weights are the bin sizes)."""
    blocks: list[list[float]] = []  # [right, n, first bin, last bin]
    for i, (n, right) in enumerate(counts):
        if n == 0:
            continue
        blocks.append([right, n, i, i])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            last = blocks.pop()
            blocks[-1][0] += last[0]
            blocks[-1][1] += last[1]
            blocks[-1][3] = last[3]
    out: list[float | None] = [None] * len(counts)
    for right, n, first, last in blocks:
        for i in range(int(first), int(last) + 1):
            if counts[i][0] > 0:
                out[i] = (right + 1) / (n + 2)
    return out


class Calibration:
    def __init__(
        self,
        counts: Mapping[str, Sequence[tuple[int, int]]] | None = None,
        *,
        edges: Sequence[float] = DEFAULT_EDGES,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        source: str = "",
    ) -> None:
        edges = tuple(float(e) for e in edges)
        if len(edges) < 2 or edges[0] != 0.0 or edges[-1] != 1.0 or any(b <= a for a, b in itertools.pairwise(edges)):
            raise ValueError("edges must increase from 0.0 to 1.0")
        if type(min_samples) is not int or min_samples < 1:
            raise ValueError("min_samples must be a whole number of at least 1")
        self.edges, self.min_samples, self.source = edges, min_samples, source
        self._counts: dict[str, tuple[tuple[int, int], ...]] = {}
        for key, rows in (counts or {}).items():
            rows = tuple((int(n), int(right)) for n, right in rows)
            if len(rows) != len(edges) - 1 or any(n < 0 or not 0 <= right <= n for n, right in rows):
                raise ValueError(f"the table for {key!r} does not match the bins")
            self._counts[key] = rows
        self._accuracy = {key: _monotone(list(rows)) for key, rows in self._counts.items()}

    @classmethod
    def identity(cls) -> Calibration:
        """No table: every raw confidence is taken at face value, and reported as not calibrated."""
        return cls()

    @property
    def fitted(self) -> bool:
        return bool(self._counts)

    @classmethod
    def fit(
        cls,
        samples: Iterable[Sample],
        *,
        edges: Sequence[float] = DEFAULT_EDGES,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        source: str = "",
    ) -> Calibration:
        edges = tuple(edges)
        counts: dict[str, list[list[int]]] = {}
        for s in samples:
            if not math.isfinite(s.raw):
                continue
            i = _bin(edges, s.raw)
            for key in (_key(s.tier, s.route), _key(s.tier, POOLED)):
                row = counts.setdefault(key, [[0, 0] for _ in range(len(edges) - 1)])[i]
                row[0] += 1
                row[1] += bool(s.right)
        return cls(
            {k: [tuple(r) for r in rows] for k, rows in counts.items()},
            edges=edges,
            min_samples=min_samples,
            source=source,
        )

    def apply(self, tier: str, route: str, raw: float) -> tuple[float, bool]:
        """(confidence, whether a table was used). A thin or missing bin returns the raw number, not calibrated."""
        if not math.isfinite(raw):
            return 0.0, False
        raw = min(max(raw, 0.0), 1.0)
        i = _bin(self.edges, raw)
        for key in (_key(tier, route), _key(tier, POOLED)):
            rows = self._counts.get(key)
            value = self._accuracy.get(key, [None] * (len(self.edges) - 1))[i]
            if rows is not None and rows[i][0] >= self.min_samples and value is not None:
                return value, True
        return raw, False

    # -- data ------------------------------------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": VERSION,
            "source": self.source,
            "edges": list(self.edges),
            "min_samples": self.min_samples,
            "counts": {k: [list(r) for r in rows] for k, rows in sorted(self._counts.items())},
        }

    @classmethod
    def from_dict(cls, data: object) -> Calibration:
        if not isinstance(data, dict) or data.get("version") != VERSION:
            raise ValueError("not a routing reliability table of this version")
        try:
            return cls(data["counts"], edges=data["edges"], min_samples=data["min_samples"], source=str(data.get("source", "")))
        except (KeyError, TypeError) as error:
            raise ValueError("the reliability table is malformed") from error

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        """Read a table from a file. Raises `ValueError`; the caller words it for the user."""
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read the reliability table {Path(path).name}") from error

    def dump(self, path: str | Path) -> None:
        """Write the table as JSON, one line per table so a diff of two runs reads."""
        data = self.to_dict()
        head = {k: v for k, v in data.items() if k != "counts"}
        lines = [json.dumps({k: v}, ensure_ascii=False)[1:-1] for k, v in sorted(head.items())]
        rows = [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in data["counts"].items()]
        body = ",\n".join([*(f" {line}" for line in lines), ' "counts": {\n' + ",\n".join(rows) + "\n }"])
        Path(path).write_text("{\n" + body + "\n}\n", encoding="utf-8")


def expected_calibration_error(pairs: Iterable[tuple[float, bool]], edges: Sequence[float] = DEFAULT_EDGES) -> float:
    """Weighted mean gap between confidence and accuracy over the bins. 0 is perfectly calibrated; None-case is 0."""
    bins: dict[int, list[float]] = {}
    total = 0
    for confidence, right in pairs:
        row = bins.setdefault(_bin(edges, confidence), [0.0, 0.0, 0.0])
        row[0] += 1
        row[1] += confidence
        row[2] += bool(right)
        total += 1
    if not total:
        return 0.0
    return sum(abs(row[1] / row[0] - row[2] / row[0]) * row[0] for row in bins.values()) / total


__all__ = ["DEFAULT_EDGES", "DEFAULT_MIN_SAMPLES", "Calibration", "Sample", "expected_calibration_error"]
