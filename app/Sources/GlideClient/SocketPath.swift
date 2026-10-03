import Foundation

/// Where the core's socket lives. Never a TCP port: the transport is a Unix domain socket owned by this user.
public enum SocketPath {
    /// The environment variable that overrides the default.
    public static let environmentName = "GLIDE_SOCKET"
    /// sockaddr_un.sun_path is 104 bytes on macOS, including the terminating NUL.
    public static let maxLength = 103

    /// Order: `--socket <path>` argument, then `GLIDE_SOCKET`, then a per-user default.
    public static func resolve(arguments: [String], environment: [String: String], home: String = NSHomeDirectory()) -> String {
        if let i = arguments.firstIndex(of: "--socket"), arguments.indices.contains(i + 1) {
            return expand(arguments[i + 1], home: home)
        }
        if let value = environment[environmentName], !value.isEmpty { return expand(value, home: home) }
        return home + "/Library/Application Support/Glide/glide.sock"
    }

    static func expand(_ path: String, home: String) -> String {
        if path == "~" { return home }
        if path.hasPrefix("~/") { return home + String(path.dropFirst(1)) }
        return path
    }

    public enum Problem: Error, Equatable, Sendable {
        case empty, tooLong(Int), missing, notASocket, notOwnedByCurrentUser
    }

    /// Refuses a path that is not a socket this user owns, so the app never sends what the user says to a socket
    /// another account could have put there.
    public static func validate(_ path: String, currentUser: uid_t = getuid()) -> Problem? {
        if path.isEmpty { return .empty }
        if path.utf8.count > maxLength { return .tooLong(path.utf8.count) }
        var info = stat()
        guard lstat(path, &info) == 0 else { return .missing }
        guard (info.st_mode & S_IFMT) == S_IFSOCK else { return .notASocket }
        guard info.st_uid == currentUser else { return .notOwnedByCurrentUser }
        return nil
    }
}
