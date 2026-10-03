import Darwin
import Foundation

/// A connected Unix domain socket. Reads on a dedicated thread, writes on a serial queue, and closes the
/// descriptor only after both are done with it.
public final class UnixSocketTransport: LineTransport, @unchecked Sendable {
    public let incoming: AsyncThrowingStream<Data, Error>
    private let continuation: AsyncThrowingStream<Data, Error>.Continuation
    private let fd: Int32
    private let writeQueue = DispatchQueue(label: "glide.socket.write")
    private let writeLock = NSLock()  // held while writing; taken before stateLock
    private let stateLock = NSLock()
    private enum State { case open, shutDown, closed }
    private var state = State.open
    private var readerDone = false

    /// Connects to the socket at `path`. Throws `TransportError`, never a raw errno.
    public static func connect(path: String) throws -> UnixSocketTransport {
        if let problem = SocketPath.validate(path) {
            // A socket that is not there yet just means the core has not started: the client keeps retrying.
            if problem == .missing { throw TransportError.connectFailed("no socket at the path yet") }
            throw TransportError.socketPathInvalid(String(describing: problem))
        }
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw TransportError.connectFailed("socket() failed: errno \(errno)") }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8)
        withUnsafeMutableBytes(of: &addr.sun_path) { raw in
            for (i, b) in bytes.enumerated() { raw[i] = b }
            raw[bytes.count] = 0
        }
        let result = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard result == 0 else {
            let code = errno
            Darwin.close(fd)
            throw TransportError.connectFailed("connect() failed: errno \(code)")
        }
        return UnixSocketTransport(adopting: fd)
    }

    /// Takes ownership of a connected socket (for example one end of `socketpair`).
    public init(adopting fd: Int32) {
        self.fd = fd
        (incoming, continuation) = AsyncThrowingStream<Data, Error>.makeStream()
        var one: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
        var timeout = timeval(tv_sec: 5, tv_usec: 0)  // bounds a write to a peer that stopped reading
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        let thread = Thread { [self] in readLoop() }
        thread.name = "glide.socket.read"
        thread.start()
    }

    private func readLoop() {
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)
        var failure: Error?
        while true {
            let n = buffer.withUnsafeMutableBytes { read(fd, $0.baseAddress, $0.count) }
            if n > 0 {
                continuation.yield(Data(buffer[0..<n]))
            } else if n == 0 {
                break
            } else if errno == EINTR {
                continue
            } else {
                stateLock.lock()
                let expected = state != .open  // a close() we asked for is not an error
                stateLock.unlock()
                if !expected { failure = TransportError.closed }
                break
            }
        }
        stateLock.lock()
        readerDone = true
        stateLock.unlock()
        continuation.finish(throwing: failure)
        close()
    }

    public func send(_ line: Data) async throws {
        let out = line + Data([0x0A])
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            writeQueue.async { [self] in
                do {
                    try writeAll(out)
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    private func writeAll(_ data: Data) throws {
        writeLock.lock()
        defer { writeLock.unlock() }
        stateLock.lock()
        let open = state == .open
        stateLock.unlock()
        guard open else { throw TransportError.closed }
        var offset = 0
        try data.withUnsafeBytes { raw in
            while offset < raw.count {
                let n = write(fd, raw.baseAddress! + offset, raw.count - offset)
                if n > 0 {
                    offset += n
                } else if n < 0 && errno == EINTR {
                    continue
                } else {
                    throw TransportError.writeFailed("write() failed: errno \(errno)")
                }
            }
        }
    }

    public func close() {
        stateLock.lock()
        if state == .open {
            state = .shutDown
            Darwin.shutdown(fd, SHUT_RDWR)  // wakes the reader and any blocked writer
        }
        stateLock.unlock()
        closeDescriptorIfIdle()
    }

    private func closeDescriptorIfIdle() {
        writeLock.lock()
        defer { writeLock.unlock() }
        stateLock.lock()
        defer { stateLock.unlock() }
        if readerDone && state != .closed {
            state = .closed
            Darwin.close(fd)
        }
    }
}
