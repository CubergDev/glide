"""Pinning a point on a synthetic desktop: what is read, what is frozen, what expires, what is refused."""

from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image
from point_fakes import Clock, SyntheticDesktop, reply_writer
from writer_endpoint import writer_for

from glide.assistant import point_ask
from glide.assistant.point_ask import HARD_TTL_S, PointStopped, PointUnavailable, capture_point
from glide.computer.platform_adapter import using
from glide.computer.point_types import PointTarget


def test_the_exact_point_is_frozen_and_text_needs_no_capture():
    fake = SyntheticDesktop()
    with using(fake), capture_point() as selected:
        fake.point = (1100, 700)
        observed = selected.observation()
        assert observed["point"] == (200, 150)
        assert observed["target"]["label"] == "錯誤 0007: connection failed"
        assert selected.image is None
    assert fake.calls == [("pointer",), ("target", (200.0, 150.0))]
    assert selected.target is None
    with pytest.raises(PointStopped):
        selected.observation()


def test_disclosure_is_required_before_any_model_call():
    with using(SyntheticDesktop()), capture_point() as selected:
        writer = SimpleNamespace(generate=lambda *args: pytest.fail("must not disclose"))
        with pytest.raises(PointUnavailable, match="Allow disclosure"):
            selected.ask(writer, "What is this?")


def test_a_text_only_model_is_never_sent_an_image():
    with using(SyntheticDesktop()), capture_point(with_image=True) as selected:
        writer = SimpleNamespace(generate=lambda *args: pytest.fail("must not send an image"))
        with pytest.raises(PointUnavailable, match="text-only"):
            selected.ask(writer, "What is this?", allow_model=True, vision=lambda: False)


@pytest.mark.parametrize("scale", [1, 2, 3])
def test_small_crop_marks_the_same_point_at_different_display_scales(scale):
    fake = SyntheticDesktop(scale=scale)
    with using(fake), capture_point((200, 150), with_image=True, radius=100) as selected:
        assert selected.region == (100, 50, 300, 250)
        assert selected.image.size == (200 * scale, 200 * scale)
        assert selected.image.getpixel((100 * scale + 9, 100 * scale)) == (255, 0, 0)
        assert selected.observation()["image_marker"] == "red ring at the selected point"
    assert fake.calls == [("target", (200, 150)), ("region", (200, 150), 100), ("target", (200, 150))]
    with pytest.raises(ValueError):
        fake.raw.getpixel((0, 0))  # the adapter's own capture was closed


def test_image_is_bounded_after_retina_resizing():
    with using(SyntheticDesktop(scale=3)), capture_point(with_image=True, radius=240) as selected:
        assert max(selected.image.size) <= 768


@pytest.mark.parametrize(
    "kwargs", [{"ttl": 0}, {"ttl": HARD_TTL_S + 1}, {"ttl": 301}, {"radius": float("inf")}, {"point": (float("nan"), 1)}]
)
def test_invalid_selections_are_refused_before_desktop_access(kwargs):
    fake = SyntheticDesktop()
    with using(fake), pytest.raises(PointUnavailable):
        capture_point(**kwargs)
    assert fake.calls == []


def test_protected_fields_are_refused_before_any_pixel_capture():
    fake = SyntheticDesktop(PointTarget("AXTextField", protected=True))
    with using(fake), pytest.raises(PointUnavailable, match="Protected"):
        capture_point(with_image=True)
    assert all(c[0] != "region" for c in fake.calls)


def test_a_changed_target_during_capture_discards_the_image():
    fake = SyntheticDesktop()
    fake.after = PointTarget("AXTextField", protected=True)
    with using(fake), pytest.raises(PointUnavailable, match="changed"):
        capture_point(with_image=True)
    with pytest.raises(ValueError):
        fake.raw.getpixel((0, 0))


def test_no_accessible_target_requires_an_explicit_image_crop():
    with using(SyntheticDesktop(None)), pytest.raises(PointUnavailable, match="No accessible"):
        capture_point()
    with using(SyntheticDesktop(None)), capture_point(with_image=True) as selected:
        assert selected.target is None and selected.image is not None


# -- age and the hard time limit (d) -------------------------------------------------------------------


def test_the_observation_says_how_old_it_is_and_the_selection_expires_at_the_hard_limit():
    clock = Clock()
    with using(SyntheticDesktop()), capture_point(clock=clock) as selected:
        assert selected.observation()["age_s"] == 0.0
        clock.advance(45.25)
        assert selected.observation()["age_s"] == 45.2 and selected.age_s() == 45.25
        clock.advance(HARD_TTL_S - 45.25 - 0.01)
        selected.check()  # one hundredth of a second short of the limit: still good
        clock.advance(0.01)
        with pytest.raises(PointStopped):
            selected.check()
        with pytest.raises(PointStopped):
            selected.observation()


def test_the_default_lifetime_is_two_minutes():
    assert HARD_TTL_S == 120
    with using(SyntheticDesktop()), capture_point(clock=Clock(0.0)) as selected:
        assert selected.expires_at == 120.0


def test_expired_and_cancelled_selections_make_no_request():
    clock = Clock()
    with using(SyntheticDesktop()), capture_point(clock=clock) as selected:
        clock.advance(HARD_TTL_S)
        with pytest.raises(PointStopped):
            selected.ask(reply_writer(), "Explain this", allow_model=True)
    with using(SyntheticDesktop()), capture_point() as selected:
        selected.cancel()
        with pytest.raises(PointStopped):
            selected.ask(reply_writer(), "Explain this", allow_model=True)


def test_cancellation_during_a_request_drops_the_late_answer_and_survives_close(monkeypatch):
    with using(SyntheticDesktop()), capture_point(with_image=True) as selected:

        def answer(writer, question, observed, image, history):
            selected.close()
            assert image.getpixel((0, 0)) == (255, 255, 255)  # the request owns its own copy of the crop
            return SimpleNamespace(text="late answer", uncertain=False)

        monkeypatch.setattr(point_ask, "compose_point_answer", answer)
        with pytest.raises(PointStopped):
            selected.ask(reply_writer(), "Explain this", allow_model=True)


@pytest.mark.parametrize("question", ["", "  ", "a" * 2049])
def test_empty_and_oversized_questions_are_refused(question):
    with using(SyntheticDesktop()), capture_point() as selected, pytest.raises(PointUnavailable, match="Question"):
        selected.ask(reply_writer(), question, allow_model=True)


def test_a_selection_reports_nothing_of_the_screen_in_its_repr():
    with using(SyntheticDesktop()), capture_point(with_image=True) as selected:
        assert "0007" not in repr(selected) and "Image" not in repr(selected)


# -- the whole path over the real chain adapter, on loopback -----------------------------------------


def test_the_answer_goes_through_the_provider_chain_to_a_loopback_endpoint(clean_env, endpoint):
    """glide.toml -> chains -> ChainWriter -> compose_point_answer -> an OpenAI-style server on this machine."""
    endpoint.state["reply"] = json.dumps({"answer": "這是錯誤 0007，請檢查連線。", "uncertain": True})  # noqa: RUF001
    fake = SyntheticDesktop(scale=2)
    with using(fake), capture_point(with_image=True) as selected:
        answer = selected.ask(writer_for(endpoint.url), "這個錯誤是什麼？", allow_model=True, vision=lambda: True)  # noqa: RUF001
    assert answer.text == "這是錯誤 0007，請檢查連線。" and answer.uncertain  # noqa: RUF001
    (request,) = endpoint.seen
    body = request["body"]
    assert body["model"] == "smart-model"  # the answer is the smart chain's job, whichever model glide.toml names
    content = body["messages"][-1]["content"]
    packet = json.loads(next(part["text"] for part in content if part["type"] == "text"))
    assert packet["question"] == "這個錯誤是什麼？" and packet["observed"]["target"]["label"].startswith("錯誤 0007")  # noqa: RUF001
    assert packet["observed"]["age_s"] >= 0.0
    image_part = next(part for part in content if part["type"] == "image_url")
    with Image.open(io.BytesIO(base64.b64decode(image_part["image_url"]["url"].split(",")[1]))) as image:
        assert max(image.size) <= 768
    assert set(body["response_format"]["json_schema"]["schema"]["properties"]) == {"answer", "uncertain"}
    assert all(c[0] in {"pointer", "target", "region"} for c in fake.calls)
