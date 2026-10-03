# Voice: live verification checklist

Nothing below can be proven offline. The test suite uses fake devices, fake transcribers and a fake clock; it
shows the logic is right, not that a microphone, a speaker or a vendor behaves. Do these once on the Mac you
will use, with the extras installed (`speech`: sounddevice, numpy, onnxruntime, and websockets only if you wire
a websockets-based adapter). Tick each box or write down what you saw. A failed box is a bug to report, not
something to tune around.

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

Expected by design: half duplex. While Glide speaks, and for `echo_tail_s` after, the microphone hears silence.

- [ ] Say a short question. Glide answers aloud. It does not answer itself, and it does not treat its own
      voice as a new request (watch for a second, unprompted turn after the answer ends).
- [ ] Talk **over** a long answer. Expected: nothing happens until it finishes. This is the known limit of
      speakers without echo cancellation, not a defect. If you want to interrupt by voice, use a headset.
- [ ] Say "stop" after the answer has ended and while a task runs: the task stops.
- [ ] After a long answer, speak right away when it ends. Your first word should not be clipped by the tail;
      if it is, lower `echo_tail_s` and note the value that works for your room.

## 2. Headset mode (`headset = true`)

- [ ] Start a long answer and talk over it. The voice cuts within about half a second of you starting, and
      your request is answered (it is not thrown away by the interruption).
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

## 6. Realtime text-to-speech adapter (optional)

Only if you register the `RealtimeElevenLabsTTS` adapter (see the report's shared edits). Its socket protocol
was ported from a snapshot and **not** checked against the vendor's documentation.

- [ ] A sentence is spoken, in the voice and language you configured, with audio starting before the
      sentence is fully generated.
- [ ] A wrong key gives an `auth` failure on screen that does not print the key, and the chain falls over
      to the next TTS provider visibly (a switch is shown, never silent).
- [ ] If the vendor rejects the connection (a changed path or message shape), report the status shown; the
      path is the constant `STREAM_PATH` in `glide/speech/elevenlabs.py`.

## Not covered even by this list

Echo cancellation (there is none), noisy rooms, Bluetooth headsets that change sample rate when the
microphone opens, many hours of continuous listening, and any vendor's behaviour under load.
