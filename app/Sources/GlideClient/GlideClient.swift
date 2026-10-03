import Foundation
import GlideProtocol

public enum ConnectionStatus: Sendable, Equatable {
    case disconnected(reason: String?)
    case connecting(attempt: Int)
    case handshaking
    case ready(CoreHello)
    /// Will not retry: the core speaks a different protocol version, or the socket path is unusable.
    case failed(String)

    public var isReady: Bool { if case .ready = self { true } else { false } }
}

public enum ClientEvent: Sendable, Equatable {
    case status(ConnectionStatus)
    case frame(CoreFrame)
    /// A line that could not be used. Carries a description of the shape problem, never the content.
    case protocolIssue(String)
}

public enum ClientError: Error, Equatable, Sendable {
    case notConnected
    /// The connection dropped before the core answered. The outcome of the request is unknown: ask again for
    /// the settings instead of repeating the change.
    case connectionLost
    case timedOut
    case unexpectedReply
    /// The core answered the request with an `error` message.
    case rejected(code: String, message: String)
}

public struct ClientConfig: Sendable {
    public var clientName = "glide-app"
    public var clientVersion = "0.1.0"
    public var handshakeTimeout: TimeInterval = 5
    public var requestTimeout: TimeInterval = 10
    /// Seconds to wait before reconnect attempt 1, 2, 3...; the last value repeats.
    public var backoff: [TimeInterval] = [0.5, 1, 2, 5, 10]

    public init() {}
}

/// What the app model needs from the core connection. `GlideClient` is the real one.
public protocol CoreLink: Sendable {
    var events: AsyncStream<ClientEvent> { get }
    func start() async
    func stop() async
    func send(_ command: AppCommand) async throws
    func settingsGet() async throws -> SettingsReply
    func settingsSet(baseRevision: Int, changes: [SettingChange]) async throws -> SettingsResult
}

/// Connects, says hello, reads frames, answers pings, and reconnects with backoff.
///
/// It never queues a command while disconnected and never replays one after a reconnect: a command whose
/// outcome is unknown is reported as such and left to the caller to reconcile.
public actor GlideClient: CoreLink {
    public typealias Connector = @Sendable () async throws -> any LineTransport

    public nonisolated let events: AsyncStream<ClientEvent>
    private let emit: AsyncStream<ClientEvent>.Continuation
    private let connector: Connector
    private let config: ClientConfig
    private var runTask: Task<Void, Never>?
    private var transport: (any LineTransport)?
    private(set) public var status: ConnectionStatus = .disconnected(reason: nil)
    private var counter = 0
    private var pending: [String: CheckedContinuation<CoreEvent, Error>] = [:]

    public init(connector: @escaping Connector, config: ClientConfig = ClientConfig()) {
        self.connector = connector
        self.config = config
        (events, emit) = AsyncStream<ClientEvent>.makeStream(bufferingPolicy: .unbounded)
    }

    /// A client for the Unix socket at `path`.
    public init(socketPath: String, config: ClientConfig = ClientConfig()) {
        self.init(connector: { try UnixSocketTransport.connect(path: socketPath) }, config: config)
    }

    public func start() {
        guard runTask == nil else { return }
        runTask = Task { await self.run() }
    }

    public func stop() {
        runTask?.cancel()
        runTask = nil
        transport?.close()
        transport = nil
        failPending(ClientError.connectionLost)
        setStatus(.disconnected(reason: "stopped"))
    }

    // MARK: Sending

    public func send(_ command: AppCommand) async throws {
        _ = try await sendFrame(command)
    }

    public func settingsGet() async throws -> SettingsReply {
        switch try await request(.settingsGet) {
        case .settings(let reply): return reply
        case .error(let e): throw ClientError.rejected(code: e.code, message: e.message)
        default: throw ClientError.unexpectedReply
        }
    }

    public func settingsSet(baseRevision: Int, changes: [SettingChange]) async throws -> SettingsResult {
        switch try await request(.settingsSet(SettingsSet(baseRevision: baseRevision, changes: changes))) {
        case .settingsResult(let result): return result
        case .error(let e): throw ClientError.rejected(code: e.code, message: e.message)
        default: throw ClientError.unexpectedReply
        }
    }

    @discardableResult
    private func sendFrame(_ command: AppCommand) async throws -> String {
        guard status.isReady, let transport else { throw ClientError.notConnected }
        let id = nextID()
        try await transport.send(try GlideCodec.encode(AppFrame(id: id, command: command)))
        return id
    }

    private func request(_ command: AppCommand) async throws -> CoreEvent {
        guard status.isReady, let transport else { throw ClientError.notConnected }
        let id = nextID()
        let line = try GlideCodec.encode(AppFrame(id: id, command: command))
        let timeout = config.requestTimeout
        return try await withCheckedThrowingContinuation { cont in
            pending[id] = cont
            Task {
                try? await Task.sleep(for: .seconds(timeout))
                self.expire(id)
            }
            Task {
                do { try await transport.send(line) } catch { self.fail(id, error) }
            }
        }
    }

    private func nextID() -> String {
        counter += 1
        return "a\(counter)"
    }

    private func expire(_ id: String) { fail(id, ClientError.timedOut) }

    private func fail(_ id: String, _ error: Error) {
        pending.removeValue(forKey: id)?.resume(throwing: error)
    }

    private func failPending(_ error: Error) {
        let all = pending
        pending = [:]
        for (_, cont) in all { cont.resume(throwing: error) }
    }

    // MARK: Connection loop

    private func setStatus(_ new: ConnectionStatus) {
        guard new != status else { return }
        status = new
        emit.yield(.status(new))
    }

    private func run() async {
        var attempt = 0
        while !Task.isCancelled {
            attempt += 1
            setStatus(.connecting(attempt: attempt))
            var reason: String?
            do {
                let t = try await connector()
                transport = t
                let outcome = await session(t)
                if case .fatal(let message) = outcome {
                    setStatus(.failed(message))
                    return
                }
                if case .dropped(let why) = outcome { reason = why }
                if case .ready = status { attempt = 0 }
            } catch let e as TransportError {
                if case .socketPathInvalid(let why) = e {
                    setStatus(.failed("socket path unusable: \(why)"))
                    return
                }
                reason = String(describing: e)
            } catch {
                reason = "connect failed"
            }
            transport?.close()
            transport = nil
            failPending(ClientError.connectionLost)
            if Task.isCancelled { return }
            setStatus(.disconnected(reason: reason))
            let wait = config.backoff.isEmpty ? 1 : config.backoff[min(max(attempt, 1), config.backoff.count) - 1]
            try? await Task.sleep(for: .seconds(wait))
        }
    }

    private enum SessionOutcome { case dropped(String?), fatal(String) }

    private func session(_ t: any LineTransport) async -> SessionOutcome {
        setStatus(.handshaking)
        do {
            try await t.send(try GlideCodec.encode(AppFrame(command: .hello(AppHello(client: config.clientName, clientVersion: config.clientVersion)))))
        } catch {
            return .dropped("could not send hello")
        }
        let watchdog = Task {
            try await Task.sleep(for: .seconds(config.handshakeTimeout))
            t.close()
        }
        defer { watchdog.cancel() }
        var framer = LineFramer()
        var handshook = false
        do {
            for try await chunk in t.incoming {
                for item in framer.feed(chunk) {
                    let line: Data
                    switch item {
                    case .tooLong(let n):
                        emit.yield(.protocolIssue(ProtocolError.lineTooLong(n).description))
                        continue
                    case .line(let l): line = l
                    }
                    let frame: CoreFrame
                    do {
                        frame = try GlideCodec.decodeCore(line)
                    } catch ProtocolError.unsupportedVersion(let v) {
                        return .fatal(ProtocolError.unsupportedVersion(v).description)
                    } catch {
                        emit.yield(.protocolIssue(String(describing: error)))
                        continue
                    }
                    if !handshook {
                        guard case .hello(let hello) = frame.event else {
                            return .fatal("core did not start with hello")
                        }
                        guard hello.protocolVersion == GlideWire.version else {
                            return .fatal(ProtocolError.unsupportedVersion(hello.protocolVersion).description)
                        }
                        handshook = true
                        watchdog.cancel()
                        setStatus(.ready(hello))
                        continue
                    }
                    await handle(frame, transport: t)
                }
            }
        } catch {
            return .dropped("read failed")
        }
        return .dropped(handshook ? "core closed the connection" : "no hello from core")
    }

    private func handle(_ frame: CoreFrame, transport t: any LineTransport) async {
        if case .ping = frame.event {
            try? await t.send(try GlideCodec.encode(AppFrame(command: .pong)))
            return
        }
        if let replyTo = frame.replyTo, let cont = pending.removeValue(forKey: replyTo) {
            cont.resume(returning: frame.event)
            return
        }
        emit.yield(.frame(frame))
    }
}
