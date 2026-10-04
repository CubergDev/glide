import Foundation
import Testing
@testable import GlideProtocol

private func json(_ data: Data) -> String { String(decoding: data, as: UTF8.self) }
private func line(_ s: String) -> Data { Data(s.utf8) }

@Suite("Wire format")
struct CodecTests {
    @Test func appHelloEncodesTheDocumentedShape() throws {
        let frame = AppFrame(command: .hello(AppHello(client: "glide-app", clientVersion: "0.1.0")))
        #expect(json(try GlideCodec.encode(frame)) == #"{"data":{"client":"glide-app","client_version":"0.1.0","protocol":1},"type":"hello","v":1}"#)
    }

    @Test func coreHelloDecodes() throws {
        let raw = #"{"v":1,"type":"hello","data":{"protocol":1,"core_version":"x","session_id":"s1","capabilities":["voice"],"recording_content":true}}"#
        let frame = try GlideCodec.decodeCore(line(raw))
        #expect(frame.event == .hello(CoreHello(coreVersion: "x", sessionId: "s1", capabilities: ["voice"], recordingContent: true)))
    }

    @Test func everyCoreEventRoundTrips() throws {
        let events: [CoreEvent] = [
            .hello(CoreHello(coreVersion: "1", sessionId: "s")),
            .state(StateEvent(assistant: .awaitingApproval, handsFree: true, muted: true, detail: "mic_unavailable")),
            .transcript(TranscriptEvent(utteranceId: "u1", role: .user, text: "hello", partial: true)),
            .transcript(TranscriptEvent(utteranceId: "u2", role: .assistant, text: nil, redacted: true)),
            .speech(SpeechEvent(utteranceId: "u1", phase: .interrupted)),
            .level(LevelEvent(source: .mic, value: 0.25)),
            .task(TaskEvent(taskId: "t1", phase: .verified, step: 3, summary: "opened settings", verified: true)),
            .task(TaskEvent(taskId: "t1", phase: .reconcileRequired)),
            .switchNotice(SwitchNotice(role: "tts", fromSlot: "a:b", toSlot: nil, kind: "timeout", reason: "slow")),
            .approvalRequest(ApprovalRequest(approvalId: "a1", taskId: "t1", kind: .input, command: "click Save", expiresInS: 30)),
            .settings(SettingsReply(revision: 4, settings: GlideSettings(roles: [
                RoleSettings(role: "stt", chain: [SlotSettings(name: "p:m", provider: "p", model: "m", keyEnv: "SOME_KEY", keyPresent: true, status: .ready)], pinned: "p:m"),
            ]))),
            .settingsResult(SettingsResult(ok: false, revision: 4, errors: [SettingError(key: "voice.silence_ms", message: "out of range")])),
            .error(ErrorEvent(code: "x", message: "y", fatal: true)),
            .ping,
        ]
        for event in events {
            let frame = CoreFrame(id: "i", replyTo: "r", event: event)
            let back = try GlideCodec.decodeCore(try GlideCodec.encode(frame))
            #expect(back == frame, "round trip failed for \(event.typeName)")
        }
    }

    @Test func everyAppCommandRoundTrips() throws {
        let commands: [AppCommand] = [
            .hello(AppHello(client: "c", clientVersion: "1")),
            .textInput(TextInput(text: "open the calendar")),
            .interrupt,
            .stop(StopRequest(taskId: "t1")),
            .stop(StopRequest()),
            .approvalResponse(ApprovalResponse(approvalId: "a1", decision: .deny)),
            .voiceControl(VoiceControl(action: .mute)),
            .settingsGet,
            .settingsSet(SettingsSet(baseRevision: 2, changes: [
                .handsFree(true), .headset(false), .silenceMs(700), .language("en"), .recordContent(false), .actEnabled(true),
                .pin(role: "tts", slot: nil), .pin(role: "stt", slot: "p:m"),
            ])),
            .pong,
        ]
        for command in commands {
            let frame = AppFrame(id: "n1", command: command)
            #expect(try GlideCodec.decodeApp(try GlideCodec.encode(frame)) == frame, "round trip failed for \(command.typeName)")
        }
    }

    @Test func unknownTypeFromANewerCoreIsIgnoredNotFatal() throws {
        let frame = try GlideCodec.decodeCore(line(#"{"v":1,"type":"future_thing","data":{"a":1}}"#))
        #expect(frame.event == .unknown(type: "future_thing"))
    }

    @Test func unknownEnumValuesFallBackInsteadOfFailing() throws {
        let state = try GlideCodec.decodeCore(line(#"{"v":1,"type":"state","data":{"assistant":"dreaming"}}"#))
        #expect(state.event == .state(StateEvent(assistant: .unknown)))
        let approval = try GlideCodec.decodeCore(line(#"{"v":1,"type":"approval_request","data":{"approval_id":"a","kind":"teleport","command":"c"}}"#))
        guard case .approvalRequest(let r) = approval.event else { Issue.record("not an approval"); return }
        #expect(r.kind == .other)
    }

    @Test func extraFieldsAreIgnored() throws {
        let f = try GlideCodec.decodeCore(line(#"{"v":1,"type":"level","extra":1,"data":{"source":"mic","value":0.5,"more":true}}"#))
        #expect(f.event == .level(LevelEvent(source: .mic, value: 0.5)))
    }

    @Test func wrongVersionIsRefused() {
        #expect(throws: ProtocolError.unsupportedVersion(2)) {
            try GlideCodec.decodeCore(line(#"{"v":2,"type":"ping"}"#))
        }
    }

    @Test func malformedLinesNameTheProblemWithoutEchoingContent() {
        let secretish = "the user said something private"
        do {
            _ = try GlideCodec.decodeCore(line(#"{"v":1,"type":"state","data":{"assistant":7,"x":"\#(secretish)"}}"#))
            Issue.record("should have thrown")
        } catch let e as ProtocolError {
            #expect(!e.description.contains(secretish))
        } catch {
            Issue.record("wrong error type")
        }
        #expect(throws: ProtocolError.self) { try GlideCodec.decodeCore(line("not json")) }
        #expect(throws: ProtocolError.self) { try GlideCodec.decodeCore(line(#"{"v":1,"type":"state"}"#)) }
    }

    @Test func appRefusesUnknownCommandType() {
        #expect(throws: ProtocolError.self) { try GlideCodec.decodeApp(line(#"{"v":1,"type":"launch_missiles"}"#)) }
    }

    @Test func encodedLinesContainNoNewlineEvenWhenTextDoes() throws {
        let data = try GlideCodec.encode(AppFrame(command: .textInput(TextInput(text: "a\nb\r\nc"))))
        #expect(!data.contains(0x0A))
        #expect(try GlideCodec.decodeApp(data).command == .textInput(TextInput(text: "a\nb\r\nc")))
    }

    @Test func oversizeLineIsRefusedOnEncode() {
        let big = String(repeating: "x", count: GlideWire.maxLineBytes + 1)
        #expect(throws: ProtocolError.self) { try GlideCodec.encode(AppFrame(command: .textInput(TextInput(text: big)))) }
    }
}

@Suite("No credentials on the wire")
struct SecretSafetyTests {
    @Test func settingsCarryKeyVariableNamesNeverValues() throws {
        let slot = SlotSettings(name: "p:m", provider: "p", model: "m", keyEnv: "MY_PROVIDER_KEY", keyPresent: true)
        let text = json(try GlideCodec.encode(CoreFrame(event: .settings(SettingsReply(revision: 1,
            settings: GlideSettings(roles: [RoleSettings(role: "r", chain: [slot])]))))))
        #expect(text.contains(#""key_env":"MY_PROVIDER_KEY""#))
        #expect(text.contains(#""key_present":true"#))
        for banned in ["api_key", "apikey", "secret", "token", "authorization", "password", "bearer"] {
            #expect(!text.lowercased().contains(banned), "\(banned) must not appear")
        }
    }

    @Test func settingChangesAreAClosedSet() throws {
        let data = line(#"{"key":"providers.openai.api_key","value":"sk-nope"}"#)
        #expect(throws: DecodingError.self) { try JSONDecoder().decode(SettingChange.self, from: data) }
        let keys = Set([
            SettingChange.handsFree(true), .headset(true), .silenceMs(1), .language("en"), .recordContent(true),
            .actEnabled(true), .pin(role: "r", slot: nil),
        ].map(\.key))
        #expect(keys == ["voice.hands_free", "voice.headset", "voice.silence_ms", "voice.language",
                         "privacy.record_content", "computer.act_enabled", "roles.pin"])
    }

    @Test func approvalDecisionsCannotExpressAlwaysAllow() {
        #expect(Set(ApprovalDecision.allCases.map(\.rawValue)) == ["approve", "deny"])
    }
}

@Suite("Line framing")
struct LineFramerTests {
    @Test func splitsLinesAndKeepsPartialTail() {
        var f = LineFramer()
        #expect(f.feed(line("{\"a\":1}\n{\"b\"")) == [.line(line("{\"a\":1}"))])
        #expect(f.pendingBytes == 4)
        #expect(f.feed(line(":2}\n\n")) == [.line(line("{\"b\":2}"))])
        #expect(f.pendingBytes == 0)
    }

    @Test func toleratesCRLFAndBlankLines() {
        var f = LineFramer()
        #expect(f.feed(line("a\r\n\r\nb\n")) == [.line(line("a")), .line(line("b"))])
    }

    @Test func oversizeLineIsDroppedOnceAndStreamRecovers() {
        var f = LineFramer(limit: 8)
        let out = f.feed(line("123456789012345\nok\n"))
        #expect(out == [.tooLong(15), .line(line("ok"))])
        #expect(f.pendingBytes == 0)
    }

    @Test func oversizeLineSpanningChunksReportsOnce() {
        var f = LineFramer(limit: 4)
        #expect(f.feed(line("123456")).isEmpty)
        #expect(f.feed(line("789")).isEmpty)
        #expect(f.feed(line("0\nz\n")) == [.tooLong(10), .line(line("z"))])
    }

    @Test func byteAtATimeDelivery() {
        var f = LineFramer()
        var lines: [Data] = []
        for b in Array("{\"x\":1}\n".utf8) {
            for case .line(let l) in f.feed(Data([b])) { lines.append(l) }
        }
        #expect(lines == [line("{\"x\":1}")])
    }
}
