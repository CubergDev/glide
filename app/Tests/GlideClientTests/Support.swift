import Foundation
import GlideClient
import GlideProtocol

/// Collects what an async stream yields, so a test can wait for a condition without owning the iterator.
final class Collector<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [T] = []
    private var task: Task<Void, Never>?

    init<S: AsyncSequence & Sendable>(_ stream: S) where S.Element == T {
        task = Task { [weak self] in
            do { for try await x in stream { self?.append(x) } } catch {}
        }
    }

    private func append(_ x: T) { lock.withLock { items.append(x) } }
    var all: [T] { lock.withLock { items } }
    func stop() { task?.cancel() }

    func wait(timeout: TimeInterval = 3, for predicate: @Sendable (T) -> Bool) async -> T? {
        let end = Date().addingTimeInterval(timeout)
        while Date() < end {
            if let hit = all.first(where: predicate) { return hit }
            try? await Task.sleep(for: .milliseconds(5))
        }
        return nil
    }
}

/// Plays the core on the other end of a transport.
final class FakeCore: @unchecked Sendable {
    let transport: any LineTransport
    private let frames: Collector<AppFrame>

    init(_ transport: any LineTransport) {
        self.transport = transport
        let (stream, cont) = AsyncStream<AppFrame>.makeStream()
        frames = Collector(stream)
        Task {
            var framer = LineFramer()
            do {
                for try await chunk in transport.incoming {
                    for case .line(let l) in framer.feed(chunk) {
                        if let f = try? GlideCodec.decodeApp(l) { cont.yield(f) }
                    }
                }
            } catch {}
            cont.finish()
        }
    }

    var received: [AppFrame] { frames.all }

    func waitFor(timeout: TimeInterval = 3, _ predicate: @escaping @Sendable (AppFrame) -> Bool) async -> AppFrame? {
        await frames.wait(timeout: timeout, for: predicate)
    }

    func send(_ event: CoreEvent, id: String? = nil, replyTo: String? = nil) async throws {
        try await transport.send(try GlideCodec.encode(CoreFrame(id: id, replyTo: replyTo, event: event)))
    }

    func sendRaw(_ text: String) async throws { try await transport.send(Data(text.utf8)) }

    func greet(version: Int = GlideWire.version, recording: Bool = false) async throws {
        try await send(.hello(CoreHello(protocolVersion: version, coreVersion: "test", sessionId: "s", recordingContent: recording)))
    }

    func close() { transport.close() }
}

func isHello(_ f: AppFrame) -> Bool { if case .hello = f.command { true } else { false } }

func fastConfig() -> ClientConfig {
    var c = ClientConfig()
    c.handshakeTimeout = 0.3
    c.requestTimeout = 1
    c.backoff = [0.05]
    return c
}

func isReady(_ e: ClientEvent) -> Bool { if case .status(.ready) = e { true } else { false } }

/// Hands out the client ends of in-memory pairs, one per connection attempt, and keeps the core ends.
final class PairFactory: @unchecked Sendable {
    private let lock = NSLock()
    private var cores: [FakeCore] = []
    var connections: Int { lock.withLock { cores.count } }
    var cores_: [FakeCore] { lock.withLock { cores } }

    func connector() -> GlideClient.Connector {
        { [self] in
            let (a, b) = InMemoryTransport.pair()
            lock.withLock { cores.append(FakeCore(b)) }
            return a
        }
    }

    func core(_ i: Int, timeout: TimeInterval = 3) async -> FakeCore? {
        let end = Date().addingTimeInterval(timeout)
        while Date() < end {
            if let c = lock.withLock({ cores.indices.contains(i) ? cores[i] : nil }) { return c }
            try? await Task.sleep(for: .milliseconds(5))
        }
        return nil
    }
}
