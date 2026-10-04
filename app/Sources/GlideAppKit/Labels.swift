import GlideClient
import GlideProtocol

// User-facing wording for states. These are interface copy, not model or provider names.

extension AssistantState {
    public var label: String {
        switch self {
        case .idle: "Ready"
        case .listening: "Listening"
        case .thinking: "Thinking"
        case .acting: "Working on your Mac"
        case .speaking: "Speaking"
        case .asking: "Needs an answer"
        case .awaitingApproval: "Waiting for your approval"
        case .error: "Something went wrong"
        case .unknown: "Busy"
        }
    }
}

extension ConnectionStatus {
    public var label: String {
        switch self {
        case .disconnected(let reason): reason.map { "Not connected (\($0))" } ?? "Not connected"
        case .connecting(let n): n > 1 ? "Connecting, attempt \(n)" : "Connecting"
        case .handshaking: "Connecting"
        case .ready: "Connected"
        case .failed(let why): "Cannot connect: \(why)"
        }
    }
}

extension ApprovalKind {
    public var label: String {
        switch self {
        case .screen: "Screen"
        case .input: "Mouse or keyboard"
        case .app: "App"
        case .url: "Web address"
        case .file: "File"
        case .network: "Network"
        case .spend: "Spends money"
        case .other: "Other"
        }
    }
}

extension TaskPhase {
    public var label: String {
        switch self {
        case .started: "Started"
        case .step: "Working"
        case .attempted: "Attempted, not yet checked"
        case .verified: "Checked"
        case .unverified: "Could not confirm the result"
        case .reconcileRequired: "Outcome unknown"
        case .completed: "Done"
        case .failed: "Failed"
        case .stopped: "Stopped"
        case .unknown: "Working"
        }
    }
}
