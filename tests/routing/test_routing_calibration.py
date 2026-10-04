"""Calibration is measured, smoothed, monotone, distrusts thin data, and survives a round trip through JSON."""

from __future__ import annotations

import json
import random

import pytest

from glide.routing import Calibration, Sample, expected_calibration_error


def samples(tier, route, raw, right, wrong):
    return [Sample(tier, route, raw, True)] * right + [Sample(tier, route, raw, False)] * wrong


def test_with_no_table_the_raw_number_comes_back_and_says_it_is_not_calibrated():
    cal = Calibration.identity()
    assert not cal.fitted and cal.apply("classifier", "execute", 0.93) == (0.93, False)


def test_a_bin_is_replaced_by_its_smoothed_measured_accuracy():
    cal = Calibration.fit(samples("classifier", "execute", 0.95, 18, 2))
    value, used = cal.apply("classifier", "execute", 0.91)
    assert used and value == pytest.approx(19 / 22)  # (right + 1) / (n + 2)


def test_a_thin_bin_is_not_trusted_it_falls_to_the_tier_pool_and_then_to_raw():
    data = samples("classifier", "execute", 0.95, 9, 1) + samples("classifier", "answer", 0.95, 2, 0)
    cal = Calibration.fit(data, min_samples=5)
    pooled, used = cal.apply("classifier", "answer", 0.95)  # 2 answers only: the pooled classifier bin has 12
    assert used and pooled == pytest.approx(12 / 14)
    assert cal.apply("classifier", "execute", 0.45) == (0.45, False)  # nothing there
    assert cal.apply("fast_llm", "execute", 0.95) == (0.95, False)  # a tier it never saw


def test_a_higher_raw_confidence_is_never_worth_less_than_a_lower_one():
    data = samples("classifier", "execute", 0.65, 9, 1) + samples("classifier", "execute", 0.95, 5, 5)
    cal = Calibration.fit(data)
    low, _ = cal.apply("classifier", "execute", 0.65)
    high, _ = cal.apply("classifier", "execute", 0.95)
    assert high >= low
    assert low == high == pytest.approx(15 / 22)  # the two bins were pooled


def test_one_point_zero_is_in_the_last_bin_and_out_of_range_is_clamped():
    cal = Calibration.fit(samples("classifier", "answer", 1.0, 10, 0))
    assert cal.apply("classifier", "answer", 1.0)[1] and cal.apply("classifier", "answer", 7.0)[1]
    assert cal.apply("classifier", "answer", float("nan")) == (0.0, False)


def test_a_table_round_trips_through_json_and_files(tmp_path):
    cal = Calibration.fit(samples("classifier", "execute", 0.85, 8, 2), source="unit test")
    again = Calibration.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert again.to_dict() == cal.to_dict() and again.source == "unit test"
    cal.dump(tmp_path / "t.json")
    assert Calibration.load(tmp_path / "t.json").apply("classifier", "execute", 0.85) == cal.apply("classifier", "execute", 0.85)


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {"version": 2},
        {"version": 1},
        {"version": 1, "edges": [0.0, 1.0], "min_samples": 5, "counts": {"a:b": [[1, 2]]}},
        {"version": 1, "edges": [0.0, 1.0], "min_samples": 5, "counts": {"a:b": [[1, 1], [2, 2]]}},
        {"version": 1, "edges": [0.0, 0.5], "min_samples": 5, "counts": {}},
        {"version": 1, "edges": [0.0, 1.0], "min_samples": 0, "counts": {}},
    ],
)
def test_a_malformed_table_is_refused(data):
    with pytest.raises(ValueError):
        Calibration.from_dict(data)


def test_a_file_that_is_not_a_table_is_refused_without_its_contents(tmp_path):
    path = tmp_path / "t.json"
    path.write_text("SECRET")
    with pytest.raises(ValueError) as error:
        Calibration.load(path)
    assert "SECRET" not in str(error.value)


def test_calibrating_an_overconfident_model_lowers_the_error_on_data_it_was_not_fitted_on():
    rng = random.Random(7)

    def draw(n):  # says 0.9 or 0.95 but is right 60% and 70% of the time: overconfident
        out = []
        for _ in range(n):
            raw = rng.choice([0.9, 0.95])
            out.append(Sample("classifier", "execute", raw, rng.random() < (0.6 if raw == 0.9 else 0.7)))
        return out

    train, test = draw(400), draw(400)
    cal = Calibration.fit(train)
    raw_pairs = [(s.raw, s.right) for s in test]
    cal_pairs = [(cal.apply(s.tier, s.route, s.raw)[0], s.right) for s in test]
    assert expected_calibration_error(raw_pairs) > 0.2
    assert expected_calibration_error(cal_pairs) < 0.08
    assert expected_calibration_error([]) == 0.0
