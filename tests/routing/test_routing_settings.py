"""`[routing]` is validated once, with unknown keys refused, and every number is a named default."""

from __future__ import annotations

import pytest

from glide.providers.config import ConfigError
from glide.routing import RoutingSettings, build_router
from glide.routing import settings as settings_module


def test_defaults_are_the_named_defaults():
    s = RoutingSettings()
    assert s.min_confidence == settings_module.DEFAULT_MIN_CONFIDENCE
    assert s.act_min_confidence == settings_module.DEFAULT_ACT_MIN_CONFIDENCE
    assert s.act_min_confidence >= s.min_confidence  # acting is never the easier call
    assert (s.memory_hints, s.speculative_fast, s.escalate_to_reason) == (False, False, True)


def test_a_table_overrides_and_an_empty_one_is_the_defaults():
    assert RoutingSettings.from_table(None) == RoutingSettings() == RoutingSettings.from_table({})
    s = RoutingSettings.from_table({"min_confidence": 0.7, "act_min_confidence": 0.9, "history_turns": 6})
    assert (s.min_confidence, s.act_min_confidence, s.history_turns) == (0.7, 0.9, 6)


def test_an_unknown_key_is_refused_and_the_known_ones_are_named():
    with pytest.raises(ConfigError, match=r"unknown key 'act_min_confidense'.*act_min_confidence"):
        RoutingSettings.from_table({"act_min_confidense": 0.9})


@pytest.mark.parametrize(
    "table",
    [
        {"min_confidence": 0.1},  # the classifier cannot report less than about a third, so the floor could never fire
        {"min_confidence": 1.5},
        {"min_confidence": "high"},
        {"min_confidence": True},
        {"min_confidence": float("nan")},
        {"min_confidence": 0.9, "act_min_confidence": 0.6},
        {"history_turns": 99},
        {"history_turns": 2.5},
        {"max_clarifications": -1},
        {"memory_hints": "yes"},
        {"speculative_fast": 1},
        {"fast_timeout_s": 0},
        {"calibration_file": 3},
    ],
)
def test_a_bad_value_is_a_config_error(table):
    with pytest.raises(ConfigError, match=r"\[routing\]"):
        RoutingSettings.from_table(table)


def test_load_reads_the_table_from_a_toml_file(tmp_path):
    path = tmp_path / "glide.toml"
    path.write_text("[routing]\nmin_confidence = 0.65\nmemory_hints = true\n[speech]\nheadset = true\n")
    s = RoutingSettings.load(path)
    assert (s.min_confidence, s.memory_hints) == (0.65, True)
    assert RoutingSettings.load(None) == RoutingSettings()


def test_load_reports_a_bad_file_without_its_contents(tmp_path):
    bad = tmp_path / "glide.toml"
    bad.write_text("[routing\nSECRET")
    with pytest.raises(ConfigError) as error:
        RoutingSettings.load(bad)
    assert "SECRET" not in str(error.value)
    with pytest.raises(ConfigError, match="cannot read"):
        RoutingSettings.load(tmp_path / "missing.toml")
    wrong = tmp_path / "wrong.toml"
    wrong.write_text("routing = 3\n")
    with pytest.raises(ConfigError, match="must be a table"):
        RoutingSettings.load(wrong)


class _Config:
    def __init__(self, classifier=None, fast=None):
        self._classifier, self._fast = classifier, fast

    def classifier(self):
        if self._classifier is None:
            raise ConfigError("no usable classifier provider.")
        return self._classifier

    def llm(self, role="fast"):
        assert role == "fast"
        if self._fast is None:
            raise ConfigError("no usable llm.fast provider.")
        return self._fast


def test_build_router_takes_the_two_chains_and_names_a_tier_it_could_not_build():
    router, missing = build_router(_Config(classifier=object(), fast=object()), table={"min_confidence": 0.7})
    assert missing == () and router.settings.min_confidence == 0.7
    router, missing = build_router(_Config(classifier=object()))
    assert missing == ("fast_llm",) and router.fast_llm is None
    router, missing = build_router(_Config())
    assert missing == ("classifier", "fast_llm") and router.route("hello").why_code == "tiers_failed"


def test_build_router_loads_the_table_named_in_the_settings(tmp_path):
    from glide.routing import Calibration, Sample

    path = tmp_path / "reliability.json"
    Calibration.fit([Sample("classifier", "answer", 0.9, True)] * 6, source="a test").dump(path)
    router, _ = build_router(_Config(classifier=object()), table={"calibration_file": str(path)})
    assert router.calibration.fitted and router.calibration.source == "a test"
    path.write_text("not json")
    with pytest.raises(ConfigError, match="calibration_file"):
        build_router(_Config(classifier=object()), table={"calibration_file": str(path)})
