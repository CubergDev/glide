import Foundation
import Testing
@testable import GlideProtocol

/// The golden lines under tests/fixtures/app_protocol are shared with the Python core's tests: the core's tests build each
/// `core/*.json` with its own encoder and compare, and parse each `app/*.json` with its own parser. These tests decode the
/// same files with the Swift codec, so the two implementations are checked against the same bytes.
private let fixtureRoot: URL = {
    // app/Tests/GlideProtocolTests/FixtureTests.swift -> the repository root is three directories up from this file's folder.
    var url = URL(fileURLWithPath: #filePath)
    for _ in 0..<4 { url.deleteLastPathComponent() }
    return url.appendingPathComponent("tests/fixtures/app_protocol")
}()

private func fixtures(_ side: String) throws -> [(name: String, data: Data)] {
    let dir = fixtureRoot.appendingPathComponent(side)
    let names = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".json") }.sorted()
    return try names.map { ($0, try Data(contentsOf: dir.appendingPathComponent($0))) }
}

@Suite("Shared fixtures")
struct FixtureTests {
    @Test func everyCoreFixtureDecodes() throws {
        let all = try fixtures("core")
        #expect(all.count >= 15)
        for (name, data) in all {
            let frame = try GlideCodec.decodeCore(data)
            if name == "approval_closed.json" {
                // A message type this app version does not know: ignored, never an error (compatibility inside version 1).
                #expect(frame.event == .unknown(type: "approval_closed"), "\(name)")
            } else if case .unknown(let type) = frame.event {
                Issue.record("\(name) decoded as an unknown type \(type)")
            }
        }
    }

    @Test func aRedactedTranscriptHasNoTextAndTheExtraFieldIsIgnored() throws {
        let data = try fixtures("core").first { $0.name == "transcript_redacted.json" }!.data
        let frame = try GlideCodec.decodeCore(data)
        #expect(frame.event == .transcript(TranscriptEvent(utteranceId: "u1", role: .user, text: nil, partial: false, redacted: true)))
    }

    @Test func theSettingsFixtureCarriesNamesAndNoKeys() throws {
        let data = try fixtures("core").first { $0.name == "settings.json" }!.data
        guard case .settings(let reply) = try GlideCodec.decodeCore(data).event else { Issue.record("not settings"); return }
        #expect(reply.revision == 3)
        #expect(reply.settings.roles.map(\.role) == ["stt", "tts"])
        #expect(reply.settings.roles[0].chain[1].keyPresent == false)
        #expect(reply.settings.roles[1].pinned == "gamma")
    }

    @Test func everyAppFixtureDecodesAndReencodesToTheSameJSON() throws {
        let all = try fixtures("app")
        #expect(all.count >= 10)
        for (name, data) in all {
            let frame = try GlideCodec.decodeApp(data)
            let again = try GlideCodec.encode(frame)
            let a = try JSONSerialization.jsonObject(with: data) as! NSDictionary
            let b = try JSONSerialization.jsonObject(with: again) as! NSDictionary
            // `stop` with an empty data object encodes its (empty) payload; `interrupt`, `settings_get` and `pong` carry none.
            #expect(a == b || name == "stop.json", "\(name)")
        }
    }
}
