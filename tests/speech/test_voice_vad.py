"""Turn detection, the loudness fallback, and the pinned model install, with no model and no network."""

from __future__ import annotations

import hashlib
from array import array

import pytest
from guards_voice import no_real_audio  # noqa: F401

from glide.speech import vad
from glide.speech.vad import (
    FRAME_BYTES,
    EnergyProbability,
    TurnDetector,
    UtteranceTooLong,
    VadError,
    install_model,
)

FRAME = bytes(FRAME_BYTES)


def loud(level: int = 3000) -> bytes:
    return array("h", [level if i % 2 == 0 else -level for i in range(FRAME_BYTES // 2)]).tobytes()


# -- the turn detector ------------------------------------------------------------------------------


def test_a_turn_starts_with_its_pre_roll_once_and_ends_after_exactly_the_configured_silence():
    detector = TurnDetector(600)
    for _ in range(10):
        assert detector.feed(FRAME, 0) == (False, b"", False)
    started, data, commit = detector.feed(FRAME, 0.9)
    assert started and len(data) == 8 * FRAME_BYTES and not commit  # the 8 frames before, and this one: not repeated
    for _ in range(18):
        assert detector.feed(FRAME, 0) == (False, FRAME, False)
    assert detector.feed(FRAME, 0) == (False, FRAME, True)  # the 19th quiet frame is 608 ms: the end
    assert detector.feed(FRAME, 0) == (False, b"", False)  # and the next frame belongs to no turn


def test_speech_inside_the_pause_restarts_the_silence_count():
    detector = TurnDetector(600)
    detector.feed(FRAME, 0.9)
    for _ in range(18):
        detector.feed(FRAME, 0)
    assert detector.feed(FRAME, 0.9) == (False, FRAME, False)  # the 19th frame is speech again: no end
    for _ in range(18):
        assert detector.feed(FRAME, 0)[2] is False
    assert detector.feed(FRAME, 0)[2] is True


def test_a_probability_between_the_thresholds_neither_starts_nor_ends_a_turn():
    detector = TurnDetector(600)
    assert detector.feed(FRAME, 0.4) == (False, b"", False)
    detector.feed(FRAME, 0.9)
    for _ in range(40):
        assert detector.feed(FRAME, 0.4)[2] is False  # 0.4 is not silence (it is not below 0.35)


def test_finish_ends_only_a_running_turn_and_a_second_finish_ends_nothing():
    detector = TurnDetector(1000)
    assert not detector.finish()
    assert detector.feed(FRAME, 0.9)[0]
    assert detector.finish() and not detector.finish()
    assert detector.feed(FRAME, 0) == (False, b"", False)


def test_finish_after_the_silence_already_ended_the_turn_commits_nothing_more():
    """The recorded double commit: a silence boundary and a manual finish 154 ms apart must be one end, not two."""
    detector = TurnDetector(600)
    detector.feed(FRAME, 0.9)
    ends = 0
    for _ in range(19):
        ends += detector.feed(FRAME, 0)[2]
    assert ends == 1
    for _ in range(5):  # 154 ms is about five frames
        assert detector.feed(FRAME, 0) == (False, b"", False)
    assert detector.finish() is False


def test_continuous_speech_is_bounded_and_the_turn_is_discarded():
    detector = TurnDetector()
    with pytest.raises(UtteranceTooLong, match="60 seconds"):
        for _ in range(1876):
            assert detector.feed(FRAME, 0.9)[2] is False
    assert not detector.active
    assert detector.feed(FRAME, 0) == (False, b"", False)


@pytest.mark.parametrize("threshold", [0, 199, 2001])
def test_a_silence_threshold_outside_200_to_2000_ms_is_refused(threshold):
    with pytest.raises(ValueError):
        TurnDetector(threshold)


# -- the loudness fallback --------------------------------------------------------------------------


def test_energy_probability_is_speech_above_the_threshold_and_follows_a_noisy_room():
    probability = EnergyProbability(threshold=500)
    assert probability(loud(3000)) == 1.0
    assert probability(loud(100)) == 0.0
    noisy = EnergyProbability(threshold=500)
    for _ in range(10):
        assert noisy(loud(450)) == 0.0  # the room itself: below the threshold, so it sets the floor
    assert noisy(loud(1200)) == 0.0  # three times the room's 450 is 1350: this is not yet speech
    assert noisy(loud(2500)) == 1.0


def test_the_turn_detector_runs_on_the_energy_fallback_with_no_extra_installed():
    probability, detector = EnergyProbability(), TurnDetector(600)
    started = commits = 0
    frames = [FRAME] * 5 + [loud()] * 10 + [FRAME] * 25
    for frame in frames:
        s, _, c = detector.feed(frame, probability(frame))
        started, commits = started + s, commits + c
    assert (started, commits) == (1, 1)


# -- the pinned model -------------------------------------------------------------------------------

MODEL = b"not a real model, just bytes with a known checksum"
DIGEST = hashlib.sha256(MODEL).hexdigest()


def test_install_writes_the_model_only_when_the_checksum_matches(tmp_path):
    seen = []
    target = tmp_path / "models" / "vad.onnx"
    install_model(target, url="https://models.example/vad.onnx", sha256=DIGEST, fetch=lambda url: seen.append(url) or MODEL)
    assert target.read_bytes() == MODEL and seen == ["https://models.example/vad.onnx"]
    assert not list(target.parent.glob("*.part"))


def test_a_download_that_does_not_match_is_not_saved(tmp_path):
    target = tmp_path / "vad.onnx"
    with pytest.raises(VadError, match="does not match"):
        install_model(target, url="https://models.example/vad.onnx", sha256=DIGEST, fetch=lambda url: b"tampered")
    assert not target.exists() and not list(tmp_path.iterdir())


def test_install_needs_https_and_a_real_checksum(tmp_path):
    with pytest.raises(VadError, match="https"):
        install_model(tmp_path / "m", url="http://models.example/vad.onnx", sha256=DIGEST, fetch=lambda url: MODEL)
    with pytest.raises(VadError, match="64 hexadecimal"):
        install_model(tmp_path / "m", url="https://models.example/vad.onnx", sha256="", fetch=lambda url: MODEL)
    assert not list(tmp_path.iterdir())


def test_silero_refuses_a_model_that_does_not_match_before_it_loads_anything(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    path = tmp_path / "vad.onnx"
    path.write_bytes(b"swapped on disk")
    with pytest.raises(VadError, match="does not match"):
        vad.Silero(path, DIGEST)


def test_silero_without_its_extra_says_which_extra(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name in ("numpy", "onnxruntime"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(VadError, match="speech extra"):
        vad.Silero(tmp_path / "vad.onnx", DIGEST)


def test_silero_frames_keep_state_and_a_64_sample_context_between_calls():
    np = pytest.importorskip("numpy")
    seen = []

    class FakeSession:
        def run(self, _, feed):
            seen.append({k: np.array(v, copy=True) for k, v in feed.items()})
            return np.array([[0.75]], dtype=np.float32), feed["state"] + 1

    silero = vad.Silero.__new__(vad.Silero)  # the model file is not available offline: only the plumbing is under test
    silero._np, silero._session = np, FakeSession()
    silero.reset()
    frame = array("h", range(FRAME_BYTES // 2)).tobytes()
    assert silero(frame) == 0.75 and silero(frame) == 0.75
    first, second = seen
    assert first["input"].shape == (1, 64 + 512) and not first["input"][:, :64].any()  # no context before the first frame
    assert np.array_equal(second["input"][:, :64], first["input"][:, -64:])  # the last 64 samples carry over
    assert second["state"].max() == 1 and int(second["sr"]) == 16000
    with pytest.raises(ValueError, match="512 samples"):
        silero(frame[:-2])
