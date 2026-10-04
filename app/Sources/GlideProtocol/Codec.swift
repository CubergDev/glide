import Foundation

/// Encodes and decodes one line. A line never contains a newline; framing is `LineFramer`'s job.
public enum GlideCodec {
    private static func encoder() -> JSONEncoder {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return e
    }

    public static func encode(_ frame: AppFrame) throws -> Data { try encodeAny(frame) }
    /// The core side of the protocol, used by tests and by anything that stands in for the core.
    public static func encode(_ frame: CoreFrame) throws -> Data { try encodeAny(frame) }

    private static func encodeAny<T: Encodable>(_ value: T) throws -> Data {
        let data: Data
        do { data = try encoder().encode(value) } catch { throw ProtocolError.encodingFailed(String(describing: type(of: error))) }
        guard data.count <= GlideWire.maxLineBytes else { throw ProtocolError.lineTooLong(data.count) }
        return data
    }

    public static func decodeCore(_ line: Data) throws -> CoreFrame { try decodeAny(CoreFrame.self, line) }
    public static func decodeApp(_ line: Data) throws -> AppFrame { try decodeAny(AppFrame.self, line) }

    private static func decodeAny<T: Decodable>(_ type: T.Type, _ line: Data) throws -> T {
        guard line.count <= GlideWire.maxLineBytes else { throw ProtocolError.lineTooLong(line.count) }
        do {
            return try JSONDecoder().decode(type, from: line)
        } catch let e as ProtocolError {
            throw e
        } catch let e as DecodingError {
            // Describe the shape problem without echoing the content that failed to decode.
            switch e {
            case .keyNotFound(let k, _): throw ProtocolError.malformed("missing field \(k.stringValue)")
            case .typeMismatch(_, let ctx), .valueNotFound(_, let ctx), .dataCorrupted(let ctx):
                let path = ctx.codingPath.map(\.stringValue).joined(separator: ".")
                throw ProtocolError.malformed("bad value at \(path.isEmpty ? "top level" : path)")
            @unknown default: throw ProtocolError.malformed("undecodable")
            }
        } catch {
            throw ProtocolError.malformed("undecodable")
        }
    }
}

/// Splits a byte stream into lines. Feed it whatever the socket returned; it keeps the partial tail.
/// A line over the limit is dropped up to its newline and reported once, so one bad line cannot grow memory.
public struct LineFramer: Sendable {
    public enum Output: Equatable, Sendable {
        case line(Data)
        case tooLong(Int)
    }

    private var buffer = Data()
    private var discarding = false
    private var discarded = 0
    private let limit: Int

    public init(limit: Int = GlideWire.maxLineBytes) { self.limit = limit }

    public mutating func feed(_ chunk: Data) -> [Output] {
        var out: [Output] = []
        for byte in chunk {
            if byte == 0x0A {
                if discarding {
                    out.append(.tooLong(discarded))
                    discarding = false
                    discarded = 0
                } else {
                    if buffer.last == 0x0D { buffer.removeLast() }
                    if !buffer.isEmpty { out.append(.line(buffer)) }
                    buffer.removeAll(keepingCapacity: true)
                }
            } else if discarding {
                discarded += 1
            } else {
                buffer.append(byte)
                if buffer.count > limit {
                    discarding = true
                    discarded = buffer.count
                    buffer.removeAll(keepingCapacity: false)
                }
            }
        }
        return out
    }

    /// Bytes held waiting for a newline.
    public var pendingBytes: Int { buffer.count }
}
