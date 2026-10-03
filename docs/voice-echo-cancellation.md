# Talking over Glide through speakers

With a headset the microphone does not hear Glide, so you can interrupt it. With speakers it does, and a voice
detector cannot tell Glide's voice from yours. Until now speaker mode was half duplex: the microphone was blanked while
Glide spoke and for `echo_tail_s` after, so nothing you said over it was heard.

Speaker mode now keeps the microphone open and takes Glide's own voice out of it. The audio device already knows what it
handed the speaker; that is the far-end reference an echo canceller needs.

## What happens, in order

1. `glide/speech/audio.py` records what the output callback handed the sound card (silence included), at 16 kHz, with a
   running sample number. Each microphone frame gets the same kind of number and is paired with the reference at its
   number, shifted by the speaker's count when the first frame arrived. The reference is therefore never older than the
   microphone, so the echo in a frame always *lags* its reference: the one thing a canceller cannot work without. A frame
   waits about one frame (32 ms) in `read()` until its reference exists.
2. `glide/speech/echo.py`: `canceller.process(near, far)` returns the frame with the echo taken out, and keeps content-free
   statistics (ERLE, the echo expected to be left, the delay).
3. `glide/speech/turns.py`: `BargeInGate` looks at every cleaned frame while Glide may be heard. A sound is the person only
   if it is voiced, louder than the echo still expected by `barge_margin_db`, at least a quarter of the microphone's
   amplitude survived cancelling, it lasts `barge_min_voiced_ms`, and the canceller has measured at least
   `barge_min_erle_db`. Then, at once: the voice is cut (`Assistant.interrupt_speech`: the queue is dropped, the epoch is
   bumped so queued speech and any late answer are dropped, the answer being written is cancelled), and then the new turn
   starts from the frames kept since 384 ms before the sound began, so the first words are in it.
4. A weaker sound (3 dB under the margin, 96 ms) opens a stop-only probe: its audio goes to the transcriber and is acted on
   only if it is a stop phrase (the router's deterministic fast path plus `stop_phrases`). Anything else is dropped unheard,
   so Glide's own transcribed echo can never become a request. One probe at a time, one second apart.
5. Self-correction ("no wait, I meant...") is unchanged: speech inside `merge_window_s` continues the same turn.

Headset mode has no canceller and no gate on echo: the voice detector and the minimum duration decide. `echo_canceller =
"none"` (or nothing installable) is the old half-duplex speaker mode, and `auto` says so on screen.

## Configuration (`[speech]`)

| key | default | meaning |
|---|---|---|
| `echo_canceller` | `"auto"` | `"webrtc"` (the `aec` extra), `"nlms"` (numpy, the `speech` extra), `"none"`, `"auto"` (the first that loads, and which) |
| `barge_min_voiced_ms` | 190 | how long you must speak over Glide (96-1000); shorter sounds are backchannel |
| `barge_margin_db` | 8.0 | how far above the echo still expected in the frame your voice must be (3-30) |
| `barge_min_erle_db` | 6.0 | no voice interruption until the canceller removes at least this much (0-30) |
| `stop_phrases` | `[]` | extra whole-utterance stop phrases; the built-in ones stay |

Install: `uv sync --extra speech --extra aec`.

## Candidates, tried on this Mac (arm64, Python 3.13)

All against the same synthetic room (speech-like far end, delayed and filtered echo path with noise), 20 s, ERLE after the
first six seconds, echo only. Synthetic signals: relative figures, not a promise about your room.

| candidate | install here | ERLE | notes |
|---|---|---|---|
| WebRTC AEC3 via the `livekit` package | one wheel (about 20 MB, Apache-2.0 per its metadata), no compiler, no system library | about 40 dB at 20-250 ms delay | finds the delay itself; ducks the person's voice in double talk (see limits). It also installs numpy. Chosen as primary |
| speexdsp (Python binding) | source only: needs `brew install speexdsp swig` and include/library paths to build; no wheel for 3.13 | about 24 dB | works; not chosen, because a clean `uv sync` cannot install it |
| numpy two-path frequency-domain filter (ours) | numpy only | 15-28 dB after 6 s, slower at the start | the fallback; covers an echo path of 384 ms (delay and tail together), a longer one is partly cancelled |
| pyaec | installed; not evaluated | | |
| speexdsp-ns | no wheel for 3.13 | | |

Native, not built: on macOS the system has a voice-processing audio unit (`VoiceProcessingIO`, also reachable as
`AVAudioInputNode.setVoiceProcessingEnabled`) that does echo cancellation, noise suppression and gain control in the
input path of a full-duplex unit, with the app's own playback as the reference. It is the natural choice for the SwiftUI
app, and it needs no canceller in Python at all: the app would play through the same unit and send already-cleaned frames.
Things to check when building it, none verified here: how it changes the output volume or ducks other audio, its
behaviour with Bluetooth devices, and that frames arrive at the rate the voice detector expects.

## What the offline tests show, and what they cannot

`tests/speech/test_voice_echo_dsp.py` (run with `--extra speech --extra aec`; it skips without them) drives the real
device, canceller, gate and voice loop through a simulated room. Over the seeds, delays, echo levels and noise levels it
sweeps:

- ERLE after convergence: WebRTC at least 30 dB (about 40 measured), numpy filter at least 12 dB (15 to 28 measured).
- Echo only: no false barge-in and no false probe in the committed linear sweep (4 seeds, 3 delays, 2 echo levels, 2 noise
  levels: 48 runs per canceller). Wider sweeps during development (8 seeds, 288 runs per canceller, linear and with a
  soft-clipping loudspeaker) gave no false interruption and one false probe (WebRTC, nonlinear). A loudspeaker that
  distorts is an echo no linear filter removes; the committed test bounds probes there and requires zero interruptions.
- A voice 9 dB over the echo at the microphone: detected within 0.6 s of starting by both cancellers, the speaker's queue
  empty at the next output block, and the first 300 ms of the voice in the turn.
- Double talk: the filter recovers its ERLE afterwards and the numpy filter passes the voice through (correlation above 0.95).

Limits, measured and not hidden:

- A voice quieter than the echo is detected less often and later. The numpy filter needs the voice within about 3 dB of the
  echo level at the microphone; WebRTC ducks a voice in double talk and, at 0 to +3 dB, detected about 8 to 9 of 12 simulated
  voices, sometimes after more than a second. In the simulation a quiet "stop" was cut through the probe in 4 of 8 cases
  against 2 of 8 for the voice alone (numpy filter, 6 dB under the echo).
- Until the canceller has measured itself (a second or two of speech, longer for the numpy filter) and reached
  `barge_min_erle_db`, nothing interrupts by voice: the first seconds of the first answer in a session are effectively half
  duplex. The filter is not reset between answers, only when the microphone is resumed.
- A sudden change of the echo path (the volume, moving the laptop) shows in the statistics within a few frames, but one
  false interruption in that moment is possible.
- A speaker that distorts (cheap laptop speakers at high volume) limits ERLE for every linear canceller.
- Reverberation outlasting 384 ms (the numpy filter) is partly left in.
- Real speech, a real microphone, a real room and the vendor's voice detector are different from this simulation. The
  simulation used a deliberately pessimistic detector that calls any non-silent frame speech, so the gate was tested alone.
  `docs/live-checks/voice.md` says what to check on the machine.
