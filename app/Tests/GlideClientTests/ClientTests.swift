import Foundation
import Testing
@testable import GlideClient
import GlideProtocol

@Suite("Client over an in-memory transport", .serialized)
struct ClientTests {
    private func connected(config: ClientConfig = fastConfig()) async throws -> (GlideClient, FakeCore, Collector<ClientEvent>, PairFactory) {
        let factory = PairFactory()
        let client = GlideClient(connector: factory.connector(), config: config)
        let events = Collector(client.events)
        await client.start()
        let core = try #require(await factory.core(0))
        _ = try #require(await core.waitFor(isHello))
        try await core.greet()
        _ = try #require(await events.wait(for: isReady))
        return (client, core, events, factory)
    }

    @Test func handshakeSendsHelloAndBecomesReady() async throws {
        let (client, core, events, _) = try await connected()
        let hello = try #require(core.received.first)
        #expect(hello.command == .hello(AppHello(client: "glide-app", clientVersion: "0.1.0")))
        #expect(await client.status.isReady)
        events.stop()
        await client.stop()
    }

    @Test func settingsGetMatchesTheReplyByID() async throws {
        let (client, core, _, _) = try await connected()
        let reply = Task { try await client.settingsGet() }
        let req = try #require(await core.waitFor { $0.command == .settingsGet })
        let id = try #require(req.id)
        let payload = SettingsReply(revision: 7, settings: GlideSettings(voice: VoiceSettings(handsFree: true)))
        try await core.send(.settings(payload), replyTo: id)
        #expect(try await reply.value == payload)
        await client.stop()
    }

    @Test func aCoreErrorAnswersTheRequestAsRejected() async throws {
        let (client, core, _, _) = try await connected()
        let result = Task { try await client.settingsSet(baseRevision: 1, changes: [.handsFree(true)]) }
        let req = try #require(await core.waitFor { if case .settingsSet = $0.command { true } else { false } })
        try await core.send(.error(ErrorEvent(code: "stale_revision", message: "reload first")), replyTo: req.id)
        do {
            _ = try await result.value
            Issue.record("expected a rejection")
        } catch {
            #expect(error as? ClientError == .rejected(code: "stale_revision", message: "reload first"))
        }
        await client.stop()
    }

    @Test func unsolicitedSettingsAreDeliveredAsEvents() async throws {
        let (client, core, events, _) = try await connected()
        try await core.send(.settings(SettingsReply(revision: 9, settings: GlideSettings())))
        let hit = await events.wait { if case .frame(let f) = $0, case .settings = f.event { true } else { false } }
        #expect(hit != nil)
        await client.stop()
    }

    @Test func answersPingWithPong() async throws {
        let (client, core, _, _) = try await connected()
        try await core.send(.ping)
        #expect(await core.waitFor { $0.command == .pong } != nil)
        await client.stop()
    }

    @Test func requestTimesOutAndDoesNotHang() async throws {
        var config = fastConfig()
        config.requestTimeout = 0.1
        let (client, _, _, _) = try await connected(config: config)
        await #expect(throws: ClientError.timedOut) { try await client.settingsGet() }
        await client.stop()
    }

    @Test func sendingBeforeReadyThrowsAndQueuesNothing() async throws {
        let factory = PairFactory()
        let client = GlideClient(connector: factory.connector(), config: fastConfig())
        await #expect(throws: ClientError.notConnected) { try await client.send(.interrupt) }
        await #expect(throws: ClientError.notConnected) { try await client.settingsGet() }
        #expect(factory.connections == 0)
    }

    @Test func unknownMessageTypesAndBadLinesDoNotBreakTheConnection() async throws {
        let (client, core, events, _) = try await connected()
        try await core.sendRaw(#"{"v":1,"type":"from_the_future","data":{}}"#)
        try await core.sendRaw("this is not json and says private words")
        try await core.send(.state(StateEvent(assistant: .listening)))
        let issue = await events.wait { if case .protocolIssue = $0 { true } else { false } }
        guard case .protocolIssue(let why) = try #require(issue) else { return }
        #expect(!why.contains("private words"))
        #expect(await events.wait { if case .frame(let f) = $0, case .state = f.event { true } else { false } } != nil)
        #expect(await client.status.isReady)
        await client.stop()
    }

    @Test func aDifferentProtocolVersionIsFatalAndNotRetried() async throws {
        let factory = PairFactory()
        let client = GlideClient(connector: factory.connector(), config: fastConfig())
        let events = Collector(client.events)
        await client.start()
        let core = try #require(await factory.core(0))
        try await core.greet(version: 99)
        let failed = await events.wait { if case .status(.failed) = $0 { true } else { false } }
        #expect(failed != nil)
        try await Task.sleep(for: .milliseconds(200))
        #expect(factory.connections == 1, "must not reconnect to an incompatible core")
        await client.stop()
    }

    @Test func aCoreThatNeverGreetsIsDroppedAndRetried() async throws {
        let factory = PairFactory()
        let client = GlideClient(connector: factory.connector(), config: fastConfig())
        await client.start()
        _ = try #require(await factory.core(1, timeout: 3), "expected a second connection attempt")
        await client.stop()
    }

    @Test func connectionLossFailsPendingRequestsAndNeverReplaysThem() async throws {
        let (client, core, events, factory) = try await connected()
        let outcome = Task { () -> Result<SettingsResult, Error> in
            do { return .success(try await client.settingsSet(baseRevision: 1, changes: [.recordContent(true)])) }
            catch { return .failure(error) }
        }
        _ = try #require(await core.waitFor { if case .settingsSet = $0.command { true } else { false } })
        core.close()
        guard case .failure(let error) = await outcome.value else { Issue.record("expected failure"); return }
        #expect(error as? ClientError == .connectionLost)

        let second = try #require(await factory.core(1))
        _ = try #require(await second.waitFor(isHello))
        try await second.greet()
        _ = try #require(await events.wait(timeout: 3) { if case .status(.ready) = $0 { true } else { false } })
        try await Task.sleep(for: .milliseconds(100))
        #expect(second.received.count == 1, "only the hello may be sent on the new connection: \(second.received.map(\.command.typeName))")
        await client.stop()
    }

    @Test func approvalResponseGoesOutAsDocumented() async throws {
        let (client, core, _, _) = try await connected()
        try await client.send(.approvalResponse(ApprovalResponse(approvalId: "a1", decision: .approve)))
        let f = try #require(await core.waitFor { if case .approvalResponse = $0.command { true } else { false } })
        #expect(f.command == .approvalResponse(ApprovalResponse(approvalId: "a1", decision: .approve)))
        await client.stop()
    }
}
