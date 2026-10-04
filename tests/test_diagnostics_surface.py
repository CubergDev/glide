"""The recording surface of the core: content and raw URLs stay out unless recording is opted into (D3)."""

from glide.computer import diagnostics
from glide.computer.diagnostics import CONTENT_OMITTED, REDACTED, Diagnostics


def test_calls_without_a_recorder_do_nothing():
    diagnostics.event("x", goal="g")
    diagnostics.exception(ValueError("v"))
    assert diagnostics.record_content() is False


def test_content_fields_and_url_paths_are_omitted_by_default(tmp_path):
    recorder = Diagnostics(tmp_path)
    with recorder.activate():
        diagnostics.event(
            "started",
            goal="buy shoes",
            endpoint="http://127.0.0.1:9222/json/version?token=abc",
            page_text="secret page",
            elapsed_s=1.5,
        )
    (item,) = recorder.events
    assert item["details"]["goal"] == CONTENT_OMITTED and item["details"]["page_text"] == CONTENT_OMITTED
    assert item["details"]["endpoint"] == "http://127.0.0.1:9222"
    assert item["details"]["elapsed_s"] == 1.5
    assert not list(tmp_path.iterdir())  # nothing is written to the run folder


def test_opt_in_keeps_content_but_never_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("GLIDE_TEST_API_KEY", "sk-live-1234567890")
    recorder = Diagnostics(tmp_path, record_content=True)
    with recorder.activate():
        assert diagnostics.record_content() is True
        diagnostics.event("x", goal="buy shoes with sk-live-1234567890", headers={"Authorization": "Bearer abcdef"})
    details = recorder.events[0]["details"]
    assert details["goal"] == f"buy shoes with {REDACTED}"
    assert details["headers"]["Authorization"] == REDACTED


def test_exception_message_is_recorded_only_with_content_recording(tmp_path):
    quiet = Diagnostics(tmp_path)
    loud = Diagnostics(tmp_path, record_content=True)
    error = OSError("could not reach https://example.test/private?q=1")
    quiet.exception(error, stage="cdp")
    loud.exception(error, stage="cdp")
    assert quiet.error == {"type": "OSError", "message": "OSError"}
    assert "example.test/private" in loud.error["message"]


def test_scrub_text_cuts_urls_and_credentials_to_one_line():
    text = diagnostics.scrub_text("failed at https://a.test/x?y=1 with api_key=abc123 \n twice")
    assert text == f"failed at https://a.test with api_key={REDACTED} twice"


def test_recorders_are_isolated_by_context(tmp_path):
    outer, inner = Diagnostics(tmp_path), Diagnostics(tmp_path, record_content=True)
    with outer.activate():
        with inner.activate():
            assert diagnostics.record_content() is True
        assert diagnostics.record_content() is False


def test_free_text_under_reason_and_observation_is_not_kept_without_recording(tmp_path):
    recorder = Diagnostics(tmp_path)
    with recorder.activate():
        diagnostics.event("plan_validation_failed", reason="secret goal", observation={"title": "bank page"}, retrying=True)
    (item,) = recorder.events
    assert item["details"]["reason"] == CONTENT_OMITTED and item["details"]["observation"] == CONTENT_OMITTED
    assert item["details"]["retrying"] is True
