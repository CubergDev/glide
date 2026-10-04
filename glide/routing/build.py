"""Assemble a `Router` from a `GlideConfig`: the classifier chain, the fast LLM chain, the settings and the table.

A role with no usable slot is not an error here (a partial setup still works): the router simply has one tier less, and
with none it answers. The omission is not silent: `Router.route` reports `tiers_failed`, and `build_router` returns the
names of the tiers it could not build so the caller can warn once (an AGENTS.md visible fallback).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from glide.providers.chain import SwitchEvent
from glide.providers.config import ConfigError

from .calibration import Calibration
from .decision import TIER_CLASSIFIER, TIER_FAST
from .router import Router
from .settings import RoutingSettings


def build_router(
    config: Any,
    *,
    table: Mapping[str, Any] | None = None,
    stop_phrases: Iterable[str] = (),
    on_event: Callable[[SwitchEvent], None] | None = None,
) -> tuple[Router, tuple[str, ...]]:
    """(router, names of the tiers that could not be built). `table` is the parsed `[routing]` table, if any."""
    settings = RoutingSettings.from_table(table)
    missing: list[str] = []
    classifier = fast = None
    try:
        classifier = config.classifier()
    except ConfigError:
        missing.append(TIER_CLASSIFIER)
    try:
        fast = config.llm("fast")
    except ConfigError:
        missing.append(TIER_FAST)
    calibration = None
    if settings.calibration_file:
        try:
            calibration = Calibration.load(settings.calibration_file)
        except ValueError as error:
            raise ConfigError(f"[routing] calibration_file: {error}") from None
    router = Router(
        classifier,
        fast,
        settings=settings,
        calibration=calibration,
        stop_phrases=stop_phrases,
        on_event=on_event,
    )
    return router, tuple(missing)
