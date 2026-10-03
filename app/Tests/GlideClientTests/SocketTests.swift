import Darwin
import Foundation
import Testing
@testable import GlideClient
import GlideProtocol

@Suite("Unix socket")
struct SocketTests {
    @Test func resolvesPathFromArgumentThenEnvironmentThenDefault() {
        #expect(SocketPath.resolve(arguments: ["app", "--socket", "/tmp/a.sock"], environment: ["GLIDE_SOCKET": "/tmp/b.sock"], home: "/h") == "/tmp/a.sock")
        #expect(SocketPath.resolve(arguments: ["app"], environment: ["GLIDE_SOCKET": "/tmp/b.sock"], home: "/h") == "/tmp/b.sock")
        #expect(SocketPath.resolve(arguments: ["app"], environment: ["GLIDE_SOCKET": "~/x.sock"], home: "/h") == "/h/x.sock")
        #expect(SocketPath.resolve(arguments: ["app"], environment: [:], home: "/h") == "/h/Library/Application Support/Glide/glide.sock")
        #expect(SocketPath.resolve(arguments: ["app", "--socket"], environment: [:], home: "/h").hasSuffix("glide.sock"))
    }

    @Test func validateRefusesWhatIsNotOurSocket() throws {
        #expect(SocketPath.validate("") == .empty)
        #expect(SocketPath.validate(String(repeating: "a", count: 200)) == .tooLong(200))
        let dir = NSTemporaryDirectory() + "glide-\(UUID().uuidString.prefix(8))"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }
        #expect(SocketPath.validate(dir + "/nope.sock") == .missing)
        let file = dir + "/plain"
        FileManager.default.createFile(atPath: file, contents: Data())
        #expect(SocketPath.validate(file) == .notASocket)
    }

    @Test func connectToAMissingSocketThrowsATransportError() {
        #expect(throws: TransportError.self) { try UnixSocketTransport.connect(path: NSTemporaryDirectory() + "glide-missing-\(UUID().uuidString.prefix(6)).sock") }
    }

    @Test func aMissingSocketIsRetriedButAWrongKindOfFileIsFatal() async throws {
        let dir = NSTemporaryDirectory() + "glide-\(UUID().uuidString.prefix(8))"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let waiting = GlideClient(socketPath: dir + "/not-yet.sock", config: fastConfig())
        let waitingEvents = Collector(waiting.events)
        await waiting.start()
        #expect(await waitingEvents.wait { if case .status(.connecting(let n)) = $0 { n >= 3 } else { false } } != nil, "keeps trying")
        #expect(waitingEvents.all.allSatisfy { if case .status(.failed) = $0 { false } else { true } })
        await waiting.stop()

        let file = dir + "/plain"
        FileManager.default.createFile(atPath: file, contents: Data())
        let wrong = GlideClient(socketPath: file, config: fastConfig())
        let wrongEvents = Collector(wrong.events)
        await wrong.start()
        #expect(await wrongEvents.wait { if case .status(.failed) = $0 { true } else { false } } != nil)
        await wrong.stop()
    }

    @Test func clientTalksToAFakeCoreOverASocketPair() async throws {
        var fds: [Int32] = [0, 0]
        try #require(socketpair(AF_UNIX, SOCK_STREAM, 0, &fds) == 0)
        let clientEnd = UnixSocketTransport(adopting: fds[0])
        let core = FakeCore(UnixSocketTransport(adopting: fds[1]))
        let client = GlideClient(connector: { clientEnd }, config: fastConfig())
        let events = Collector(client.events)
        await client.start()
        _ = try #require(await core.waitFor(isHello))
        try await core.greet()
        _ = try #require(await events.wait(for: isReady))
        try await core.send(.state(StateEvent(assistant: .speaking, handsFree: true)))
        #expect(await events.wait { if case .frame(let f) = $0, case .state(let s) = f.event, s.assistant == .speaking { true } else { false } } != nil)
        try await client.send(.textInput(TextInput(text: "hi")))
        #expect(await core.waitFor { $0.command == .textInput(TextInput(text: "hi")) } != nil)
        await client.stop()
        core.close()
    }

    @Test func realSocketFileRoundTrip() async throws {
        // A listening socket in a private temp directory; nothing leaves this process.
        let dir = "/tmp/gl-\(UUID().uuidString.prefix(6))"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let path = dir + "/s.sock"
        let server = socket(AF_UNIX, SOCK_STREAM, 0)
        try #require(server >= 0)
        defer { Darwin.close(server) }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        withUnsafeMutableBytes(of: &addr.sun_path) { raw in
            let b = Array(path.utf8)
            for (i, x) in b.enumerated() { raw[i] = x }
            raw[b.count] = 0
        }
        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(server, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        guard bound == 0 else {
            Issue.record("bind to a temp socket was refused (errno \(errno)); run this test outside a sandbox")
            return
        }
        try #require(listen(server, 1) == 0)
        #expect(SocketPath.validate(path) == nil)
        #expect(SocketPath.validate(path, currentUser: getuid() &+ 1) == .notOwnedByCurrentUser)

        let accepted = Task.detached { accept(server, nil, nil) }
        let client = GlideClient(socketPath: path, config: fastConfig())
        let events = Collector(client.events)
        await client.start()
        let fd = await accepted.value
        try #require(fd >= 0)
        let core = FakeCore(UnixSocketTransport(adopting: fd))
        _ = try #require(await core.waitFor(isHello))
        try await core.greet()
        #expect(await events.wait(for: isReady) != nil)
        await client.stop()
        core.close()
    }
}
