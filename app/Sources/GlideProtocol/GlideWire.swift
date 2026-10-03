import Foundation

/// Constants of the wire format. The protocol is specified in app/PROTOCOL.md.
public enum GlideWire {
    /// Major protocol version. Every line carries it as `"v"`. A different value is refused, not guessed at.
    public static let version = 1
    /// Longest line (without the newline) either side accepts or sends, in bytes.
    public static let maxLineBytes = 1_048_576
}

/// What can be wrong with a line. Messages never echo line content, because a line can carry what the user said.
public enum ProtocolError: Error, Equatable, Sendable {
    case unsupportedVersion(Int)
    case lineTooLong(Int)
    case malformed(String)
    case encodingFailed(String)
}

extension ProtocolError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .unsupportedVersion(let v): "unsupported protocol version \(v) (this app speaks \(GlideWire.version))"
        case .lineTooLong(let n): "line of \(n) bytes exceeds the \(GlideWire.maxLineBytes) byte limit"
        case .malformed(let why): "malformed line: \(why)"
        case .encodingFailed(let why): "could not encode message: \(why)"
        }
    }
}
