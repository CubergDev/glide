import Foundation

// Payload types for each message. Field names on the wire are snake_case and are written out in CodingKeys so
// the wire format is visible here and does not depend on an encoder strategy.
//
// Nothing in this file can carry a credential: there is no field for a key value anywhere in the protocol.
// Providers appear by the *name* of the environment variable that holds their key and by whether it is set.

// MARK: Enums

public enum AssistantState: String, LenientEnum {
    case idle, listening, thinking, acting, speaking, asking
    case awaitingApproval = "awaiting_approval"
    case error
    case unknown
    public static var fallback: AssistantState { .unknown }
}

public enum TranscriptRole: String, LenientEnum {
    case user, assistant, unknown
    public static var fallback: TranscriptRole { .unknown }
}

public enum SpeechPhase: String, LenientEnum {
    case started, finished, interrupted, unknown
    public static var fallback: SpeechPhase { .unknown }
}

public enum LevelSource: String, LenientEnum {
    case mic, speaker, unknown
    public static var fallback: LevelSource { .unknown }
}

/// Task lifecycle. `attempted` and `verified` are separate on purpose: an attempted action is not a verified
/// effect. `reconcileRequired` means a write has an unknown outcome and nothing will be replayed.
public enum TaskPhase: String, LenientEnum {
    case started, step, attempted, verified, unverified
    case reconcileRequired = "reconcile_required"
    case completed, failed, stopped
    case unknown
    public static var fallback: TaskPhase { .unknown }
}

public enum ApprovalKind: String, LenientEnum {
    case screen, input, app, url, file, network, spend, other
    public static var fallback: ApprovalKind { .other }
}

public enum ApprovalDecision: String, Codable, Sendable, Equatable, CaseIterable {
    case approve, deny
}

public enum VoiceAction: String, Codable, Sendable, Equatable, CaseIterable {
    case mute, unmute
}

public enum SlotStatus: String, LenientEnum {
    case ready, resting, skipped, unknown
    public static var fallback: SlotStatus { .unknown }
}

// MARK: Core to app

public struct CoreHello: Codable, Sendable, Equatable {
    public var protocolVersion: Int
    public var coreVersion: String
    public var sessionId: String
    public var capabilities: [String]
    /// True when the core was started with detailed recording opted in. Informational; the app never records.
    public var recordingContent: Bool

    public init(protocolVersion: Int = GlideWire.version, coreVersion: String, sessionId: String,
                capabilities: [String] = [], recordingContent: Bool = false) {
        self.protocolVersion = protocolVersion
        self.coreVersion = coreVersion
        self.sessionId = sessionId
        self.capabilities = capabilities
        self.recordingContent = recordingContent
    }

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol", coreVersion = "core_version", sessionId = "session_id"
        case capabilities, recordingContent = "recording_content"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        protocolVersion = try c.decode(Int.self, forKey: .protocolVersion)
        coreVersion = try c.decode(String.self, forKey: .coreVersion)
        sessionId = try c.decode(String.self, forKey: .sessionId)
        capabilities = try c.decodeIfPresent([String].self, forKey: .capabilities) ?? []
        recordingContent = try c.decodeIfPresent(Bool.self, forKey: .recordingContent) ?? false
    }
}

public struct StateEvent: Codable, Sendable, Equatable {
    public var assistant: AssistantState
    public var handsFree: Bool
    public var muted: Bool
    /// A short machine-readable reason, such as "mic_unavailable". Never user content.
    public var detail: String?

    public init(assistant: AssistantState, handsFree: Bool = false, muted: Bool = false, detail: String? = nil) {
        self.assistant = assistant
        self.handsFree = handsFree
        self.muted = muted
        self.detail = detail
    }

    enum CodingKeys: String, CodingKey { case assistant, handsFree = "hands_free", muted, detail }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        assistant = try c.decode(AssistantState.self, forKey: .assistant)
        handsFree = try c.decodeIfPresent(Bool.self, forKey: .handsFree) ?? false
        muted = try c.decodeIfPresent(Bool.self, forKey: .muted) ?? false
        detail = try c.decodeIfPresent(String.self, forKey: .detail)
    }
}

/// What was said, for live display only. `text` is absent when the core withholds it (`redacted`).
/// The app keeps these in memory and never writes them anywhere.
public struct TranscriptEvent: Codable, Sendable, Equatable {
    public var utteranceId: String
    public var role: TranscriptRole
    public var text: String?
    public var partial: Bool
    public var redacted: Bool

    public init(utteranceId: String, role: TranscriptRole, text: String?, partial: Bool = false, redacted: Bool = false) {
        self.utteranceId = utteranceId
        self.role = role
        self.text = text
        self.partial = partial
        self.redacted = redacted
    }

    enum CodingKeys: String, CodingKey { case utteranceId = "utterance_id", role, text, partial, redacted }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        utteranceId = try c.decode(String.self, forKey: .utteranceId)
        role = try c.decode(TranscriptRole.self, forKey: .role)
        text = try c.decodeIfPresent(String.self, forKey: .text)
        partial = try c.decodeIfPresent(Bool.self, forKey: .partial) ?? false
        redacted = try c.decodeIfPresent(Bool.self, forKey: .redacted) ?? false
    }
}

public struct SpeechEvent: Codable, Sendable, Equatable {
    public var utteranceId: String?
    public var phase: SpeechPhase

    public init(utteranceId: String? = nil, phase: SpeechPhase) {
        self.utteranceId = utteranceId
        self.phase = phase
    }

    enum CodingKeys: String, CodingKey { case utteranceId = "utterance_id", phase }
}

/// An audio level between 0 and 1 for the waveform. Sent at a modest rate, and only while a source is active.
public struct LevelEvent: Codable, Sendable, Equatable {
    public var source: LevelSource
    public var value: Double

    public init(source: LevelSource, value: Double) {
        self.source = source
        self.value = value
    }
}

public struct TaskEvent: Codable, Sendable, Equatable {
    public var taskId: String
    public var phase: TaskPhase
    public var step: Int?
    /// A short label the core chose. Untrusted data: shown as plain text, never interpreted. The core leaves
    /// out captured content unless detailed recording is on.
    public var summary: String?
    /// Only meaningful for `verified` and `unverified`: whether a fresh observation confirmed the effect.
    public var verified: Bool?

    public init(taskId: String, phase: TaskPhase, step: Int? = nil, summary: String? = nil, verified: Bool? = nil) {
        self.taskId = taskId
        self.phase = phase
        self.step = step
        self.summary = summary
        self.verified = verified
    }

    enum CodingKeys: String, CodingKey { case taskId = "task_id", phase, step, summary, verified }
}

/// A provider switch. Mirrors the core's `SwitchEvent`. Switches are always shown, never silent.
public struct SwitchNotice: Codable, Sendable, Equatable {
    public var role: String
    public var fromSlot: String
    public var toSlot: String?
    public var kind: String
    public var reason: String

    public init(role: String, fromSlot: String, toSlot: String?, kind: String, reason: String) {
        self.role = role
        self.fromSlot = fromSlot
        self.toSlot = toSlot
        self.kind = kind
        self.reason = reason
    }

    enum CodingKeys: String, CodingKey { case role, fromSlot = "from_slot", toSlot = "to_slot", kind, reason }
}

/// Asks the user to approve exactly one command. Approval covers that command once, never the kind of command.
public struct ApprovalRequest: Codable, Sendable, Equatable, Identifiable {
    public var approvalId: String
    public var taskId: String?
    public var kind: ApprovalKind
    /// The exact command the core wants to run. Untrusted text: shown verbatim.
    public var command: String
    /// Seconds the core waits before treating silence as a denial.
    public var expiresInS: Double?

    public var id: String { approvalId }

    public init(approvalId: String, taskId: String? = nil, kind: ApprovalKind, command: String, expiresInS: Double? = nil) {
        self.approvalId = approvalId
        self.taskId = taskId
        self.kind = kind
        self.command = command
        self.expiresInS = expiresInS
    }

    enum CodingKeys: String, CodingKey {
        case approvalId = "approval_id", taskId = "task_id", kind, command, expiresInS = "expires_in_s"
    }
}

public struct ErrorEvent: Codable, Sendable, Equatable {
    public var code: String
    /// Never contains keys, headers or request bodies.
    public var message: String
    public var fatal: Bool

    public init(code: String, message: String, fatal: Bool = false) {
        self.code = code
        self.message = message
        self.fatal = fatal
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        code = try c.decode(String.self, forKey: .code)
        message = try c.decode(String.self, forKey: .message)
        fatal = try c.decodeIfPresent(Bool.self, forKey: .fatal) ?? false
    }

    enum CodingKeys: String, CodingKey { case code, message, fatal }
}

// MARK: App to core

public struct AppHello: Codable, Sendable, Equatable {
    public var protocolVersion: Int
    public var client: String
    public var clientVersion: String

    public init(protocolVersion: Int = GlideWire.version, client: String, clientVersion: String) {
        self.protocolVersion = protocolVersion
        self.client = client
        self.clientVersion = clientVersion
    }

    enum CodingKeys: String, CodingKey { case protocolVersion = "protocol", client, clientVersion = "client_version" }
}

public struct TextInput: Codable, Sendable, Equatable {
    public var text: String
    public init(text: String) { self.text = text }
}

/// Stop the current task, or one named task. Distinct from `interrupt`, which only cuts speech off.
public struct StopRequest: Codable, Sendable, Equatable {
    public var taskId: String?
    public init(taskId: String? = nil) { self.taskId = taskId }
    enum CodingKeys: String, CodingKey { case taskId = "task_id" }
}

public struct ApprovalResponse: Codable, Sendable, Equatable {
    public var approvalId: String
    public var decision: ApprovalDecision
    public init(approvalId: String, decision: ApprovalDecision) {
        self.approvalId = approvalId
        self.decision = decision
    }
    enum CodingKeys: String, CodingKey { case approvalId = "approval_id", decision }
}

public struct VoiceControl: Codable, Sendable, Equatable {
    public var action: VoiceAction
    public init(action: VoiceAction) { self.action = action }
}
