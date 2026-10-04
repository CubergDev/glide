# Voice: live verification checklist

Nothing below can be proven offline. The test suite uses fake devices, fake transcribers and a fake clock; it
shows the logic is right, not that a microphone, a speaker or a vendor behaves. Do these once on the Mac you
will use, with the extras installed (`speech`: sounddevice, numpy, onnxruntime; `aec`: livekit, for WebRTC echo
cancellation). Tick each box or write down what you saw.
A failed box is a bug to report, not something to tune around.

Setup once: `[speech]` table and providers in `glide.toml` (see `glide/speech/settings.py` for every key), the
keys exported under the variable names the file names, and, for Silero, the model file fetched with
`glide.speech.vad.install_model(path, url=..., sha256=...)` using the address and checksum you chose. Glide
ships neither: a model location without a checksum is refused.

## 0. Before anything is heard

- [ ] `glide doctor` shows an STT, a TTS and an LLM provider that can be reached (or the reason they cannot).
- [ ] Starting the voice loop asks macOS for microphone access once. Deny it: the loop must report the
      failure and not hang. Allow it and start again.
- [ ] With `vad = "silero"` and a wrong checksum the start fails with a clear message (no fallback).
      With `vad = "auto"` and the same wrong checksum it starts on loudness and says so on screen.

## 1. Speaker mode (default, `headset = false`)

Design: the microphone stays open while Glide speaks, Glide's own voice is taken out of it (`echo_canceller`, default
`auto`), and a voice over it interrupts. The old half-duplex behaviour is `echo_canceller = "none"`, or what you get when
no canceller can be built (`auto` says so on screen). See `docs/voice-echo-cancellation.md`.

- [ ] Start with `echo_canceller = "auto"`. The screen says which canceller was taken, or that none is available. Note it.
- [ ] Say a short question. Glide answers aloud. It does not answer itself, and it does not treat its own voice as a new
      request (watch for a second, unprompted turn after the answer ends). If it does, raise `echo_tail_s` and note
      the value.
- [ ] After a long answer, speak right away when it ends. Your first word should not be clipped.

### 1a. Watch the numbers

While trying the rest of this section, show the debug counters (numbers and thresholds, never what was said):

```python
# in the session that runs the voice loop (see glide/speech/session.py): one line a second
from glide.speech.session import watch

loop.start()
watch(loop)
```

A line looks like `echo: webrtc erle 31.4 dB delay 128 ms latency 18 ms reference 1481 residual 34 expected 67 underruns 0
| barge-in: confirmed 1 probes 0 candidates 3 rejected 2 waiting-for-erle 41 of 812 frames | needs 192 ms, 8.0 dB over the
echo, erle 6.0 dB (probe 96 ms, 5.0 dB)`. What to look at:

- `erle`: how many dB quieter the cleaned signal is than the microphone while Glide speaks. `?` until a second or two of
  speech has been heard. WebRTC should settle at 25 dB or more within a few seconds; the numpy filter at 12 dB or more, in
  five or six. Below `barge_min_erle_db` (6) nothing interrupts by voice: note how long that takes after you start the
  loop, and whether it ever drops again (volume or position changes will do it).
- `delay`: the echo path as the device pairs it, to a frame (32 ms). Expect your real output-plus-input delay (tens of ms)
  plus 32 ms. Over about 350 ms the numpy filter cannot cover it; report the number.
- `reference`, `residual`, `expected`: levels (16-bit scale) of what the speaker is handed, of what is left after
  cancelling, and of the echo the gate expects to be left. A voice has to reach `margin_db` above `expected`. If
  `residual` regularly exceeds `expected` while only Glide speaks, the gate is too optimistic for your room.
- `underruns`: frames that went to the canceller without their reference (the speaker stalled). Should stay 0.
- `confirmed`: interruptions by voice. `candidates`/`rejected`: sounds that came near and were dropped (backchannel, echo
  bursts). `probes`: stop-only listens. `waiting-for-erle`: voiced frames ignored because the canceller was not ready.

Checks:

- [ ] **Echo only, no false interruptions.** Ask for several long answers and stay silent and still, at three speaker
      volumes (low, normal, loud). `confirmed` stays 0 and `probes` stays 0 over a few minutes of speech. If Glide
      interrupts itself, write down `erle`, `residual`, `expected` and the volume, and raise `barge_margin_db` to see if it
      stops; that is a finding, not a fix.
- [ ] **Talk over a long answer.** At normal speaking distance say "wait, what time is it?" over it. The voice cuts within
      about half a second of you starting, `confirmed` goes up by one, and the request is answered. The transcript
      (printed, or `glide status`) starts at "wait", not at "what". Note your distance and the volume.
- [ ] **A quiet "stop".** Say "stop" quietly over the answer. It stops within about a second; `probes` goes up. Say
      "stop" while a task runs: the task stops.
- [ ] **Backchannel.** Say "mm-hm", cough, tap the desk while it speaks: it keeps going and `rejected` may go up.
- [ ] **Correction.** With `merge_window_s = 1.5`, interrupt with "open Safari" ... pause 1 s ... "no, Chrome": one request,
      both halves in the transcript, the router acts on the correction.
- [ ] **Change the volume while it speaks.** Turn the speaker up sharply mid-answer. Known limit: a false interruption in
      the next few seconds is possible while the canceller re-adapts. Note whether it happened and how long `erle` took to recover.
- [ ] **Lid, desk, position.** Move the laptop or lean away. `erle` may dip and recover. Note it.
- [ ] **A bad canceller is visible.** Set `echo_canceller = "webrtc"` without the `aec` extra installed: the start fails
      with a message, it does not fall back quietly. With `"auto"` and nothing installable you get a line saying speaker mode
      is half duplex, and talking over Glide does nothing (the old behaviour).
- [ ] **Say "stop" after the answer has ended** and while a task runs: the task stops.

## 2. Headset mode (`headset = true`)

- [ ] Start a long answer and talk over it. The voice cuts within about a quarter of a second of you speaking for
      `barge_min_voiced_ms`, and your request is answered (it is not thrown away by the interruption).
- [ ] Say "stop" over the answer: it stops and nothing more is said.
- [ ] No echo of Glide's voice is picked up as speech (a headset that leaks into its microphone will fail this).

## 3. Turn-taking and self-correction

- [ ] A pause of about half a second inside a sentence does not split it. A pause of `silence_ms` ends the turn.
- [ ] With `merge_window_s = 1.5`, say "open Safari" ... pause 1 s ... "no, Chrome". One request is made
      (check `glide status` or the printed transcript): the transcript contains both halves, and the router
      acts on the correction. Without the merge window, expect two requests and the second replaces the first.
- [ ] Say two requests a few seconds apart. Both are heard and answered; neither is dropped.
- [ ] Time from the end of your speech to the first word of the answer: note it. Each turn opens its own
      transcription connection, so this is the number that a persistent session would improve.
- [ ] Speak for more than 60 seconds without stopping: the turn is discarded with a message and nothing is
      acted on.

## 4. Idle and the microphone

- [ ] With `idle_s = 30`, do nothing for 30 seconds: the loop says the microphone is off, and the macOS
      microphone indicator goes out (the stream is closed, not only ignored).
- [ ] Resume (however the front end calls `VoiceLoop.resume()`): the indicator comes back and the first
      thing you say is heard from the start.
- [ ] Idle never fires while Glide is speaking or a computer task is running.
- [ ] Unplug or switch the input device while listening: the loop reports a microphone fault and stops
      instead of acting on a half-heard command.

## 5. What is spoken

- [ ] Run a computer task that stalls or fails. Glide says one plain sentence ("I got stuck and stopped"),
      never step counts, effect counts, run folder names or "N remain".
- [ ] A reply containing markdown, a link or a list is read as plain words.

## Not covered even by this list

Noisy rooms, Bluetooth headsets that change sample rate when the microphone opens (the reference is converted from the
output rate, which a changed rate would break: report it), many hours of continuous listening, clock drift between a
microphone and a speaker on different devices (a slowly falling `erle` over a long session is the sign), and any vendor's
behaviour under load. Echo cancellation was proved only against simulated rooms (`tests/speech/test_voice_echo_dsp.py`);
the real microphone-to-speaker path is what this file is for.
