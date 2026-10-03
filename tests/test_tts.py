"""Text to speech: HTTP streaming, the macOS `say` fallback, and the facade's failover. No network, no sound.

HTTP adapters talk to `httpx.MockTransport`. `MacSayTTS` talks to a fake `subprocess.run` that inspects the
command it was given and writes a real (tiny) WAVE file where `-o` points, so the read-back code runs for
real while `say` itself never does.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import traceback
import wave

import httpx
import pytest

from glide.providers import tts
from glide.providers.base import ProviderSpec, SpeechAudio
from glide.providers.chain import Chain, Slot
from glide.providers.errors import AllProvidersFailed, ProviderError
from glide.providers.tts import (
    TTS,
    ElevenLabsTTS,
    MacSayTTS,
    OpenAICompatTTS,
    SpeechChunk,
    build_client,
    parse_voice_list,
)

KEY = "xi-secret-key-0123456789"
ODD_CHUNKS = [b"\x01", b"\x02\x03", b"\x04\x05\x06", b"\x07"]  # no chunk ends where a sample does


# -- HTTP helpers -----------------------------------------------------------------------------------


def http(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def audio_handler(*chunks: bytes, seen: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, content=iter(chunks))

    return handler


def failing(status: int, body=b"", headers=None):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=iter([body]) if body else b"", headers=headers)

    return handler


def raising(exc: Exception):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


def eleven(handler, **kw) -> ElevenLabsTTS:
    kw.setdefault("voice", "voice-1")
    return ElevenLabsTTS("eleven", "model-x", KEY, client=http(handler), **kw)


def openai(handler, **kw) -> OpenAICompatTTS:
    return OpenAICompatTTS("oai", "tts-model", KEY, base_url="https://speech.example.test/v1", client=http(handler), **kw)


def caught_error(call) -> ProviderError:
    with pytest.raises(ProviderError) as caught:
        call()
    return caught.value


def everything_said_about(error: BaseException) -> str:
    """Every way an error is shown: str, repr, args, and the whole traceback with its chain."""
    return " ".join([str(error), repr(error), repr(error.args), "".join(traceback.format_exception(error))])


# -- ElevenLabs -------------------------------------------------------------------------------------


def test_elevenlabs_streams_pcm_with_the_documented_request():
    seen: list[httpx.Request] = []
    client = eleven(audio_handler(b"\x01\x02", b"\x03\x04", seen=seen))
    assert list(client.stream("Hello there.")) == [b"\x01\x02", b"\x03\x04"]
    (request,) = seen
    assert request.method == "POST"
    assert request.url.path == "/v1/text-to-speech/voice-1/stream"
    assert request.url.host == "api.elevenlabs.io"
    assert dict(request.url.params) == {"output_format": "pcm_24000"}  # no deprecated or opt-in query by default
    assert request.headers["xi-api-key"] == KEY
    assert KEY not in str(request.url)
    assert json.loads(request.content) == {"text": "Hello there.", "model_id": "model-x"}  # no language_code by default
    assert client.sample_rate == 24000


def test_elevenlabs_optional_parameters_are_sent_only_when_configured():
    seen: list[httpx.Request] = []
    client = eleven(
        audio_handler(b"\x01\x02", seen=seen),
        output_format="pcm_16000",
        optimize_streaming_latency=3,
        language_codes={"en": "en"},
        voice_settings={"stability": 0.5},
        voices={"yue": "voice-yue"},
    )
    list(client.stream("Hi.", language="en-US"))
    list(client.stream("Hi.", language="yue"))  # no code configured for yue: none is sent
    first, second = seen
    assert dict(first.url.params) == {"output_format": "pcm_16000", "optimize_streaming_latency": "3"}
    assert json.loads(first.content)["language_code"] == "en"
    assert json.loads(first.content)["voice_settings"] == {"stability": 0.5}
    assert "language_code" not in json.loads(second.content)
    assert second.url.path == "/v1/text-to-speech/voice-yue/stream"  # voice by language
    assert client.sample_rate == 16000


def test_elevenlabs_sends_no_model_id_when_none_is_configured():
    seen: list[httpx.Request] = []
    client = ElevenLabsTTS("eleven", "", KEY, voice="v", client=http(audio_handler(b"\x01\x02", seen=seen)))
    list(client.stream("Hi."))
    assert "model_id" not in json.loads(seen[0].content)


def test_a_voice_id_cannot_change_the_path():
    seen: list[httpx.Request] = []
    list(eleven(audio_handler(b"\x01\x02", seen=seen)).stream("Hi.", voice="a/b?c=d"))
    assert seen[0].url.raw_path.startswith(b"/v1/text-to-speech/a%2Fb%3Fc%3Dd/stream")


def test_an_explicit_voice_beats_the_language_voice_and_the_default():
    seen: list[httpx.Request] = []
    client = eleven(audio_handler(b"\x01\x02", seen=seen), voices={"en": "en-voice"})
    list(client.stream("Hi.", voice="explicit", language="en"))
    list(client.stream("Hi.", language="en-GB"))
    list(client.stream("Hi.", language="fr"))
    assert [r.url.path.split("/")[3] for r in seen] == ["explicit", "en-voice", "voice-1"]


def test_chunks_are_yielded_as_they_arrive_not_after_the_body_is_read():
    produced: list[bytes] = []

    def body():
        for part in (b"\x01\x02", b"\x03\x04", b"\x05\x06"):
            produced.append(part)
            yield part

    client = eleven(lambda request: httpx.Response(200, content=body()))
    chunks = client.stream("Hi.")
    assert next(chunks) == b"\x01\x02"
    assert produced == [b"\x01\x02"]  # the server had only been asked for the first piece
    assert list(chunks) == [b"\x03\x04", b"\x05\x06"]


@pytest.mark.parametrize("make", [eleven, openai], ids=["elevenlabs", "openai"])
def test_every_chunk_holds_whole_samples_and_no_bytes_are_lost(make):
    chunks = list(make(audio_handler(*ODD_CHUNKS)).stream("Hi."))
    assert all(len(c) % 2 == 0 for c in chunks)
    assert b"".join(chunks) == b"\x01\x02\x03\x04\x05\x06"  # the final lone byte is half a sample: dropped


def test_synthesize_collects_the_stream_into_speech_audio():
    audio = eleven(audio_handler(b"\x01\x02", b"\x03\x04")).synthesize("Hi.")
    assert audio == SpeechAudio(b"\x01\x02\x03\x04", 24000)


def test_synthesize_timeout_bounds_the_whole_call():
    ticks = iter([0.0, 5.0, 5.0, 5.0])
    client = ElevenLabsTTS(
        "eleven", "m", KEY, voice="v", client=http(audio_handler(b"\x01\x02", b"\x03\x04")), clock=lambda: next(ticks)
    )
    assert caught_error(lambda: client.synthesize("Hi.", timeout=1.0)).kind == "timeout"


def test_blank_text_sends_nothing_and_makes_no_audio():
    calls: list[httpx.Request] = []
    client = eleven(audio_handler(b"\x01\x02", seen=calls))
    assert list(client.stream("  \n")) == []
    assert client.synthesize("").pcm == b""
    assert calls == []


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, "auth"),
        (403, "auth"),
        (404, "unsupported"),
        (422, "bad_request"),
        (429, "rate_limit"),
        (500, "server"),
        (504, "timeout"),
    ],
)
def test_a_non_200_streaming_reply_is_read_and_mapped(status, kind):
    body = json.dumps({"detail": {"status": "voice_trouble", "message": "the reason the server gave"}}).encode()
    error = caught_error(lambda: list(eleven(failing(status, body)).stream("Hi.")))
    assert error.kind == kind
    assert error.status == status
    assert "the reason the server gave" in str(error)  # the body was read, not left on the wire
    assert error.provider == "eleven"


def test_a_rate_limit_carries_retry_after():
    error = caught_error(lambda: list(eleven(failing(429, b"{}", {"retry-after": "7"})).stream("Hi.")))
    assert error.kind == "rate_limit" and error.retry_after == 7.0


def test_an_error_body_is_bounded():
    error = caught_error(lambda: list(eleven(failing(500, b"x" * 100_000)).stream("Hi.")))
    assert len(str(error)) < 400


def test_a_validation_error_keeps_where_and_why_but_never_the_text_it_echoes():
    body = json.dumps(
        {"detail": [{"loc": ["body", "text"], "msg": "too long", "type": "string_too_long", "input": "PRIVATE" + " SENTENCE"}]}
    ).encode()
    private = "PRIVATE" + " SENTENCE"  # built, so the traceback's own source lines cannot contain it
    error = caught_error(lambda: list(eleven(failing(422, body)).stream(private)))
    assert "body.text: too long" in str(error)
    assert private not in everything_said_about(error)


def test_text_a_server_echoes_back_is_hidden_unless_it_is_too_short_to_hide_meaningfully():
    long_text = "Please read this private sentence aloud."
    error = caught_error(lambda: list(eleven(failing(500, f"cannot say: {long_text}".encode())).stream(long_text)))
    assert "cannot say: ***" in str(error)
    short = caught_error(lambda: list(eleven(failing(500, b"No. voice found for No.")).stream("No.")))
    assert "No. voice found" in str(short)


def test_transport_failures_become_provider_errors_without_the_request():
    refused = caught_error(lambda: list(eleven(raising(httpx.ConnectError(f"refused for {KEY}"))).stream("Hi.")))
    assert refused.kind == "transport"
    assert KEY not in everything_said_about(refused)  # not even through the chained cause
    slow = caught_error(lambda: list(eleven(raising(httpx.ReadTimeout("slow"))).stream("Hi.")))
    assert slow.kind == "timeout"


def test_a_stream_that_dies_after_audio_started_raises_after_the_audio_it_gave():
    def handler(request):
        def body():
            yield b"\x01\x02"
            raise httpx.ReadError("connection dropped")

        return httpx.Response(200, content=body())

    chunks = eleven(handler).stream("Hi.")
    assert next(chunks) == b"\x01\x02"
    assert caught_error(lambda: next(chunks)).kind == "transport"


def test_a_200_with_no_audio_is_unusable_content():
    assert caught_error(lambda: list(eleven(audio_handler()).stream("Hi."))).kind == "content"
    assert caught_error(lambda: list(eleven(audio_handler(b"\x01")).stream("Hi."))).kind == "content"  # half a sample


def test_without_a_voice_or_key_nothing_is_sent_and_the_error_lets_the_chain_move_on():
    calls: list[httpx.Request] = []
    no_voice = ElevenLabsTTS("eleven", "m", KEY, client=http(audio_handler(b"\x01\x02", seen=calls)))
    assert caught_error(lambda: list(no_voice.stream("Hi."))).kind == "unsupported"
    for key in ("", "   ", "has a space", "café"):
        client = ElevenLabsTTS("eleven", "m", key, voice="v", client=http(audio_handler(b"\x01\x02", seen=calls)))
        error = caught_error(lambda client=client: list(client.stream("Hi.")))
        assert error.kind == "auth"
        assert key.strip() not in str(error) or not key.strip()
    assert calls == []


def test_a_key_with_a_trailing_newline_is_cleaned_before_it_is_sent():
    seen: list[httpx.Request] = []
    client = ElevenLabsTTS("eleven", "m", KEY + "\n", voice="v", client=http(audio_handler(b"\x01\x02", seen=seen)))
    list(client.stream("Hi."))
    assert seen[0].headers["xi-api-key"] == KEY


def test_a_key_never_appears_in_an_error_a_repr_or_a_traceback_even_when_the_server_echoes_it():
    echo = json.dumps({"detail": {"message": f"Invalid API key {KEY}"}}).encode()
    for make, handler in (
        (eleven, failing(401, echo)),
        (openai, failing(401, json.dumps({"error": {"message": f"bad {KEY}"}}).encode())),
    ):
        client = make(handler)
        error = caught_error(lambda client=client: list(client.stream("Hi.")))
        assert error.kind == "auth"
        assert KEY not in everything_said_about(error)
        assert KEY not in repr(client) and KEY not in str(client)
    plain = caught_error(lambda: list(eleven(failing(401, f"nope {KEY}".encode())).stream("Hi.")))  # not JSON
    assert KEY not in everything_said_about(plain) and "***" in str(plain)


# -- OpenAI-compatible ------------------------------------------------------------------------------


def test_openai_compat_posts_to_audio_speech_asking_for_pcm():
    seen: list[httpx.Request] = []
    client = openai(audio_handler(b"\x01\x02", b"\x03\x04", seen=seen))
    assert list(client.stream("Hello.")) == [b"\x01\x02", b"\x03\x04"]
    (request,) = seen
    assert str(request.url) == "https://speech.example.test/v1/audio/speech"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in str(request.url)
    assert json.loads(request.content) == {"model": "tts-model", "input": "Hello.", "voice": "alloy", "response_format": "pcm"}
    assert client.sample_rate == 24000  # what OpenAI documents for pcm


def test_openai_compat_rate_voice_and_extras_are_configuration():
    seen: list[httpx.Request] = []
    client = openai(
        audio_handler(b"\x01\x02", seen=seen),
        sample_rate=16000,
        voice="nova",
        voices={"yue": "cantonese-voice"},
        instructions="Speak slowly.",
        speed=1.1,
    )
    list(client.stream("Hi."))
    list(client.stream("Hi.", language="yue"))
    assert json.loads(seen[0].content)["voice"] == "nova"
    assert json.loads(seen[0].content)["instructions"] == "Speak slowly."
    assert json.loads(seen[0].content)["speed"] == 1.1
    assert json.loads(seen[1].content)["voice"] == "cantonese-voice"
    assert client.sample_rate == 16000


def test_openai_compat_without_a_key_sends_no_authorization_header():
    seen: list[httpx.Request] = []
    client = OpenAICompatTTS(
        "local", "m", "", base_url="http://localhost:9/v1", client=http(audio_handler(b"\x01\x02", seen=seen))
    )
    list(client.stream("Hi."))
    assert "authorization" not in seen[0].headers


def test_openai_compat_error_mapping_and_synthesize():
    body = json.dumps({"error": {"message": "model is unknown", "type": "invalid_request_error"}}).encode()
    error = caught_error(lambda: openai(failing(404, body)).synthesize("Hi."))
    assert (error.kind, error.status) == ("unsupported", 404)
    assert "model is unknown" in str(error)
    assert openai(audio_handler(b"\x01\x02")).synthesize("Hi.") == SpeechAudio(b"\x01\x02", 24000)


# -- macOS say --------------------------------------------------------------------------------------

LISTING = """\
Alex                en_US    # Most people recognize me by my voice.
Eddy (English (US)) en_US    # Hello, my name is Eddy.
Samantha            en_US    # Hello, my name is Samantha.
Sinji               zh_HK    # \u4f60\u597d\uff01\u6211\u53eb\u5584\u6021\u3002
Tingting            zh_CN    # \u4f60\u597d\uff0c\u6211\u53eb\u5a77\u5a77\u3002
"""
SAY_PCM = b"\x01\x00\x02\x00\x03\x00\x04\x00" * 5000  # 40000 bytes


def wav_bytes(pcm: bytes, rate: int = 22050, channels: int = 1, width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


class FakeSay:
    """Stands in for `subprocess.run`: lists voices for `-v ?`, writes a WAVE file where `-o` points."""

    def __init__(self, listing: str = LISTING, wav: bytes | None = None, returncode: int = 0, stderr: bytes = b"", raises=None):
        self.listing = listing
        self.wav = wav_bytes(SAY_PCM) if wav is None else wav
        self.returncode, self.stderr, self.raises = returncode, stderr, raises
        self.calls: list[dict] = []
        self.written: list[str] = []

    def __call__(self, argv, input=None, capture_output=False, timeout=None, check=False, **extra):
        assert not extra, f"unexpected subprocess options {extra}"
        assert isinstance(argv, list) and argv[0] == "say"  # a list, never a shell string
        self.calls.append({"argv": argv, "input": input, "timeout": timeout})
        if self.raises:
            raise self.raises
        if argv[1:] == ["-v", "?"]:
            return subprocess.CompletedProcess(argv, 0, stdout=self.listing.encode(), stderr=b"")
        if self.returncode == 0 and self.wav is not None:
            target = argv[argv.index("-o") + 1]
            self.written.append(target)
            with open(target, "wb") as out:
                out.write(self.wav)
        return subprocess.CompletedProcess(argv, self.returncode, stdout=b"", stderr=self.stderr)

    def syntheses(self) -> list[dict]:
        return [c for c in self.calls if c["argv"][1:] != ["-v", "?"]]

    def voices_asked_for(self) -> list[str]:
        return [c["argv"][c["argv"].index("-v") + 1] for c in self.syntheses()]


def mac(fake: FakeSay, **kw) -> MacSayTTS:
    return MacSayTTS("say", runner=fake, **kw)


def test_say_synthesizes_to_a_file_and_reads_the_samples_back():
    fake = FakeSay()
    audio = mac(fake).synthesize("Hello.", language="en")
    assert audio == SpeechAudio(SAY_PCM, 22050)
    (call,) = fake.syntheses()
    argv = call["argv"]
    assert "-o" in argv  # synthesis to a file: nothing is spoken through the speakers
    assert "--file-format=WAVE" in argv and "--data-format=LEI16@22050" in argv
    assert argv[argv.index("-f") + 1] == "-" and call["input"] == b"Hello."  # the text goes on stdin, not in argv
    assert "Hello." not in argv


def test_say_never_runs_without_an_output_file_and_removes_it_afterwards():
    fake = FakeSay()
    client = mac(fake)
    client.synthesize("One.", language="en")
    client.synthesize("Two.", language="yue")
    assert len(fake.syntheses()) == 2
    assert all("-o" in c["argv"] for c in fake.syntheses())
    assert fake.written and not any(os.path.exists(p) for p in fake.written)  # the temporary files are gone


def test_say_text_that_looks_like_an_option_or_is_chinese_reaches_say_intact_on_stdin():
    fake = FakeSay()
    mac(fake).synthesize("-v Alex 你好", language="yue")
    assert fake.syntheses()[0]["input"] == "-v Alex 你好".encode()
    assert fake.voices_asked_for() == ["Sinji"]


@pytest.mark.parametrize(
    ("language", "voice"),
    [
        ("yue", "Sinji"),
        ("zh-HK", "Sinji"),
        ("zh", "Tingting"),
        ("zh-CN", "Tingting"),
        ("en", "Samantha"),
        ("en-US", "Samantha"),
        (None, "Samantha"),
    ],
)
def test_say_picks_the_voice_for_the_language(language, voice):
    fake = FakeSay()
    mac(fake).synthesize("Hi.", language=language)
    assert fake.voices_asked_for() == [voice]


def test_say_voices_are_configurable_and_an_explicit_voice_wins():
    fake = FakeSay()
    client = mac(fake, voices={"en": "Alex"})
    client.synthesize("Hi.", language="en")
    client.synthesize("Hi.", language="en", voice="eddy (english (us))")  # matched ignoring case, run under its listed name
    assert fake.voices_asked_for() == ["Alex", "Eddy (English (US))"]


def test_a_missing_voice_is_unsupported_and_no_other_voice_is_used():
    fake = FakeSay(listing="\n".join(line for line in LISTING.splitlines() if not line.startswith("Sinji")))
    client = mac(fake)
    error = caught_error(lambda: client.synthesize("你好", language="yue"))
    assert error.kind == "unsupported"
    assert "Sinji" in str(error)
    assert fake.syntheses() == []  # not even attempted in Tingting or any other voice
    assert caught_error(lambda: list(client.stream("你好", language="yue"))).kind == "unsupported"
    client.synthesize("Hi.", language="en")  # the voices that are installed still work


def test_a_language_with_no_voice_and_an_explicit_voice_that_is_not_installed_are_unsupported():
    fake = FakeSay()
    client = mac(fake)
    assert caught_error(lambda: client.synthesize("Bonjour.", language="fr")).kind == "unsupported"
    assert caught_error(lambda: client.synthesize("Hi.", voice="Nobody")).kind == "unsupported"
    assert fake.syntheses() == []


def test_the_voice_list_is_read_once_and_again_only_after_a_miss():
    fake = FakeSay()
    client = mac(fake)
    client.synthesize("Hi.", language="en")
    client.synthesize("Hi.", language="en")
    assert len(fake.calls) - len(fake.syntheses()) == 1
    fake.listing += "Daniel              en_GB    # Hello, my name is Daniel.\n"  # installed while running
    client.synthesize("Hi.", voice="Daniel")
    assert len(fake.calls) - len(fake.syntheses()) == 2
    assert fake.voices_asked_for()[-1] == "Daniel"


def test_an_unreadable_voice_list_is_unusable_content():
    assert caught_error(lambda: mac(FakeSay(listing="")).synthesize("Hi.", language="en")).kind == "content"


def test_say_failures_map_to_provider_errors():
    failed = caught_error(lambda: mac(FakeSay(returncode=1, stderr=b"Voice `X' not found\n")).synthesize("Hi.", language="en"))
    assert failed.kind == "server" and "exited 1" in str(failed) and "not found" in str(failed)
    assert caught_error(lambda: mac(FakeSay(raises=FileNotFoundError("say"))).synthesize("Hi.")).kind == "unsupported"
    timed = FakeSay(raises=subprocess.TimeoutExpired("say", 1))
    assert caught_error(lambda: mac(timed).synthesize("Hi.")).kind == "timeout"


@pytest.mark.parametrize(
    "wav",
    [
        wav_bytes(SAY_PCM, rate=16000),
        wav_bytes(SAY_PCM, channels=2),
        wav_bytes(b"\x01" * 100, width=1),
        b"FORM....AIFF not a wave file",
        b"",
    ],
    ids=["wrong rate", "stereo", "8-bit", "not a WAVE file", "empty file"],
)
def test_a_file_that_is_not_mono_16_bit_at_the_asked_rate_is_refused(wav):
    assert caught_error(lambda: mac(FakeSay(wav=wav)).synthesize("Hi.", language="en")).kind == "content"


def test_say_exiting_without_writing_a_file_is_unusable_content():
    class Quiet(FakeSay):
        def __call__(self, argv, **kw):
            if argv[1:] == ["-v", "?"]:
                return super().__call__(argv, **kw)
            return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    assert caught_error(lambda: mac(Quiet()).synthesize("Hi.", language="en")).kind == "content"


def test_say_streams_whole_sample_chunks_that_add_up_to_the_audio():
    chunks = list(mac(FakeSay(), chunk_bytes=9999).stream("Hi.", language="en"))
    assert len(chunks) > 1 and all(len(c) % 2 == 0 for c in chunks)
    assert b"".join(chunks) == SAY_PCM


def test_say_runs_nothing_for_blank_text_and_honours_rate_and_sample_rate():
    fake = FakeSay(wav=wav_bytes(SAY_PCM, rate=16000))
    client = mac(fake, sample_rate=16000, rate_wpm=180)
    assert client.synthesize("   ").pcm == b"" and fake.calls == []
    assert client.synthesize("Hi.", language="en").sample_rate == 16000
    argv = fake.syntheses()[0]["argv"]
    assert "--data-format=LEI16@16000" in argv and argv[argv.index("-r") + 1] == "180"


def test_say_uses_subprocess_run_unless_a_runner_is_given(monkeypatch):
    fake = FakeSay()
    monkeypatch.setattr(subprocess, "run", fake)
    assert MacSayTTS("say").synthesize("Hi.", language="en").pcm == SAY_PCM
    assert fake.calls


def test_an_unpatched_say_is_stopped_by_the_machine_guard_not_turned_into_a_provider_error():
    with pytest.raises(RuntimeError, match="real machine"):
        MacSayTTS("say").synthesize("Hi.", language="en")  # conftest refuses Popen; nothing ran


def test_the_voice_list_parser_handles_names_with_spaces_and_parentheses():
    assert parse_voice_list(LISTING) == [
        ("Alex", "en_US"),
        ("Eddy (English (US))", "en_US"),
        ("Samantha", "en_US"),
        ("Sinji", "zh_HK"),
        ("Tingting", "zh_CN"),
    ]
    assert parse_voice_list("garbage\n\nSomeone # no locale here\n") == []


# -- The facade -------------------------------------------------------------------------------------


def two_slots(eleven_client, say_client) -> Chain:
    return Chain("tts", [Slot("elevenlabs:model-x", eleven_client), Slot("macos_say", say_client)])


def test_a_dead_primary_fails_over_to_say_before_the_first_chunk_and_the_rate_follows():
    say = FakeSay()
    chain = two_slots(eleven(raising(httpx.ConnectError("refused"))), mac(say))
    chunks = list(TTS(chain).stream("Hello.", language="en"))
    assert b"".join(c.pcm for c in chunks) == SAY_PCM
    assert {c.sample_rate for c in chunks} == {22050}  # say's rate, not ElevenLabs'
    (event,) = chain.events
    assert (event.from_slot, event.to_slot, event.kind) == ("elevenlabs:model-x", "macos_say", "transport")
    assert say.voices_asked_for() == ["Samantha"]


def test_a_server_error_also_fails_over_and_a_healthy_primary_is_used_with_its_own_rate():
    chain = two_slots(eleven(failing(503, b"{}")), mac(FakeSay()))
    assert {c.sample_rate for c in TTS(chain).stream("Hi.", language="en")} == {22050}
    assert chain.events[-1].kind == "server"

    say = FakeSay()
    chain = two_slots(eleven(audio_handler(b"\x01\x02", b"\x03\x04")), mac(say))
    chunks = list(TTS(chain).stream("Hi.", language="en"))
    assert chunks == [SpeechChunk(b"\x01\x02", 24000), SpeechChunk(b"\x03\x04", 24000)]
    assert say.calls == [] and not chain.events


def test_a_chunk_unpacks_as_audio_and_rate():
    pcm, rate = next(TTS(two_slots(eleven(audio_handler(b"\x01\x02")), mac(FakeSay()))).stream("Hi."))
    assert (pcm, rate) == (b"\x01\x02", 24000)


def test_after_the_first_chunk_a_failure_is_raised_not_hidden_by_a_fallback():
    def handler(request):
        def body():
            yield b"\x01\x02"
            raise httpx.ReadError("dropped")

        return httpx.Response(200, content=body())

    say = FakeSay()
    chunks = TTS(two_slots(eleven(handler), mac(say))).stream("Hi.", language="en")
    assert next(chunks) == SpeechChunk(b"\x01\x02", 24000)
    assert caught_error(lambda: next(chunks)).kind == "stream"
    assert say.calls == []


def test_synthesize_fails_over_and_carries_the_rate_of_the_slot_that_answered():
    chain = two_slots(eleven(failing(500)), mac(FakeSay()))
    assert TTS(chain).synthesize("Hi.", language="en") == SpeechAudio(SAY_PCM, 22050)
    assert chain.events[-1].from_slot == "elevenlabs:model-x"
    healthy = two_slots(eleven(audio_handler(b"\x01\x02")), mac(FakeSay()))
    assert TTS(healthy).synthesize("Hi.") == SpeechAudio(b"\x01\x02", 24000)


def test_a_request_the_primary_calls_malformed_is_raised_and_not_sent_to_say():
    say = FakeSay()
    error = caught_error(lambda: list(TTS(two_slots(eleven(failing(422, b"{}")), mac(say))).stream("Hi.", language="en")))
    assert error.kind == "bad_request" and say.calls == []  # the chain's rule: our own fault fails everywhere


def test_when_the_primary_and_the_fallback_both_fail_every_error_is_reported():
    say = FakeSay(listing="\n".join(line for line in LISTING.splitlines() if not line.startswith("Sinji")))
    chain = two_slots(eleven(failing(401, b"{}")), mac(say))
    with pytest.raises(AllProvidersFailed) as caught:
        list(TTS(chain).stream("你好", language="yue"))
    assert [(name, e.kind) for name, e in caught.value.errors] == [("elevenlabs:model-x", "auth"), ("macos_say", "unsupported")]
    assert say.syntheses() == []  # Cantonese was never read in another voice


class Scripted:
    """A client whose stream gives a script per call: bytes to yield, or an exception to raise."""

    def __init__(self, sample_rate: int, *script):
        self.name, self.model, self.sample_rate = f"fake{sample_rate}", "m", sample_rate
        self.script = list(script)
        self.texts: list[str] = []

    def stream(self, text, *, voice=None, language=None):
        self.texts.append(text)
        step = self.script[min(len(self.texts), len(self.script)) - 1]
        if isinstance(step, BaseException):
            raise step
        yield from step

    def synthesize(self, text, *, voice=None, language=None, timeout=None):
        raise NotImplementedError


def test_speak_sentences_plays_in_order_and_reads_the_next_sentence_only_when_asked():
    pulled: list[str] = []

    def sentences():
        for s in ("One.", "   ", "Two.", "Three."):
            pulled.append(s)
            yield s

    client = Scripted(24000, [b"\x01\x02", b"\x03\x04"])
    spoken = TTS(Chain("tts", [Slot("a", client)])).speak_sentences(sentences())
    assert next(spoken) == SpeechChunk(b"\x01\x02", 24000)
    assert pulled == ["One."]  # the model could still be writing the rest
    one, two = SpeechChunk(b"\x01\x02", 24000), SpeechChunk(b"\x03\x04", 24000)
    assert list(spoken) == [two, one, two, one, two]  # the rest of "One.", then "Two.", then "Three."
    assert client.texts == ["One.", "Two.", "Three."]  # in order, the blank one skipped
    assert pulled == ["One.", "   ", "Two.", "Three."]


def test_the_rate_can_change_between_sentences_after_a_failover():
    primary = Scripted(24000, [b"\x01\x02"], ProviderError("down", kind="transport"))
    fallback = Scripted(22050, [b"\x09\x0a"])
    chain = Chain("tts", [Slot("primary", primary), Slot("fallback", fallback)])
    chunks = list(TTS(chain).speak_sentences(["First.", "Second."]))
    assert chunks == [SpeechChunk(b"\x01\x02", 24000), SpeechChunk(b"\x09\x0a", 22050)]
    assert (chain.events[-1].from_slot, chain.events[-1].to_slot) == ("primary", "fallback")


def test_a_provider_dying_mid_sentence_raises_to_the_caller_of_speak_sentences():
    def dies_midway():
        yield b"\x01\x02"
        raise ProviderError("dropped", kind="transport")

    class Dying(Scripted):
        def stream(self, text, *, voice=None, language=None):
            yield from dies_midway()

    spoken = TTS(Chain("tts", [Slot("a", Dying(24000)), Slot("b", Scripted(22050, [b"\x05\x06"]))])).speak_sentences(["Hi."])
    assert next(spoken) == SpeechChunk(b"\x01\x02", 24000)
    assert caught_error(lambda: next(spoken)).kind == "stream"


# -- build_client -----------------------------------------------------------------------------------


def test_build_client_dispatches_on_kind_and_options_override_the_spec():
    eleven_spec = ProviderSpec("eleven", "elevenlabs", options={"voice": "v-spec", "output_format": "pcm_16000"})
    client = build_client(eleven_spec, "model-x", KEY, {"output_format": "pcm_22050", "voices": {"yue": "v-yue"}})
    assert isinstance(client, ElevenLabsTTS)
    assert (client.name, client.model, client.sample_rate) == ("eleven", "model-x", 22050)

    oai = build_client(ProviderSpec("oai", "openai_compat", "http://localhost:9/v1", options={"sample_rate": 16000}), "m", "")
    assert isinstance(oai, OpenAICompatTTS) and oai.sample_rate == 16000

    default_oai = build_client(ProviderSpec("oai", "openai_compat"), "m", KEY)
    assert default_oai.sample_rate == 24000

    say = build_client(ProviderSpec("local", "macos_say", options={"voices": {"yue": "Sinji"}}), "", "ignored", {"rate": 150})
    assert isinstance(say, MacSayTTS) and say.sample_rate == 22050


def test_build_client_refuses_an_unknown_kind_and_bad_configuration():
    with pytest.raises(ValueError, match="elevenlabs, openai_compat or macos_say"):
        build_client(ProviderSpec("x", "typesafe"), "m", KEY)
    with pytest.raises(ValueError):
        build_client(ProviderSpec("x", "elevenlabs"), "m", KEY, {"output_format": "mp3_44100_128"})  # not raw PCM
    with pytest.raises(ValueError):
        build_client(ProviderSpec("x", "openai_compat"), "m", KEY, {"sample_rate": "fast"})


def test_a_built_client_never_shows_its_key():
    for spec in (ProviderSpec("e", "elevenlabs"), ProviderSpec("o", "openai_compat"), ProviderSpec("m", "macos_say")):
        client = build_client(spec, "m", KEY)
        assert KEY not in repr(client) and KEY not in str(client)


def test_the_module_keeps_protocol_details_in_one_constants_block():
    assert tts.ELEVENLABS_KEY_HEADER == "xi-api-key"
    assert tts.OPENAI_PCM_RATE == 24000 and tts.OPENAI_SPEECH_PATH == "/audio/speech"
    assert tts.SAY_DEFAULT_VOICES["yue"] == "Sinji" and tts.SAY_DEFAULT_VOICES["en"] == "Samantha"
