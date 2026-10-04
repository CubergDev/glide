import Foundation

/// A duplex byte stream. `UnixSocketTransport` is the real one; `InMemoryTransport` is for tests and previews.
public protocol LineTransport: Sendable {
    /// Raw chunks as they arrive. Finishes when the peer closes, or throws if the read fails.
    var incoming: AsyncThrowingStream<Data, Error> { get }
    /// Writes one line. The transport appends the newline.
    func send(_ line: Data) async throws
    /// Closes both directions. Safe to call more than once.
    func close()
}

public enum TransportError: Error, Equatable, Sendable {
    case closed
    case socketPathInvalid(String)
    case connectFailed(String)
    case writeFailed(String)
}

/// Two connected in-memory ends, so a test can play the core against a real `GlideClient`.
public final class InMemoryTransport: LineTransport, @unchecked Sendable {
    public let incoming: AsyncThrowingStream<Data, Error>
    private let inbox: AsyncThrowingStream<Data, Error>.Continuation
    private weak var peer: InMemoryTransport?
    private let lock = NSLock()
    private var closed = false

    private init() {
        (incoming, inbox) = AsyncThrowingStream<Data, Error>.makeStream()
    }

    public static func pair() -> (InMemoryTransport, InMemoryTransport) {
        let a = InMemoryTransport(), b = InMemoryTransport()
        a.peer = b
        b.peer = a
        return (a, b)
    }

    public func send(_ line: Data) async throws {
        let isClosed = lock.withLock { closed }
        guard !isClosed, let peer else { throw TransportError.closed }
        peer.inbox.yield(line + Data([0x0A]))
    }

    public func close() {
        let already = lock.withLock { let was = closed; closed = true; return was }
        guard !already else { return }
        inbox.finish()
        peer?.inbox.finish()
    }
}
