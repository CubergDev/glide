import Foundation

/// Keys of the envelope every line shares: `{"v":1,"type":"...","id":"...","reply_to":"...","data":{...}}`.
enum FrameKeys: String, CodingKey {
    case v, type, id, replyTo = "reply_to", data
}

// MARK: Core to app

public enum CoreEvent: Sendable, Equatable {
    case hello(CoreHello)
    case state(StateEvent)
    case transcript(TranscriptEvent)
    case speech(SpeechEvent)
    case level(LevelEvent)
    case task(TaskEvent)
    case switchNotice(SwitchNotice)
    case approvalRequest(ApprovalRequest)
    case settings(SettingsReply)
    case settingsResult(SettingsResult)
    case error(ErrorEvent)
    case ping
    /// A type this app does not know (a newer core). Ignored, never an error.
    case unknown(type: String)

    public var typeName: String {
        switch self {
        case .hello: "hello"
        case .state: "state"
        case .transcript: "transcript"
        case .speech: "speech"
        case .level: "level"
        case .task: "task"
        case .switchNotice: "switch"
        case .approvalRequest: "approval_request"
        case .settings: "settings"
        case .settingsResult: "settings_result"
        case .error: "error"
        case .ping: "ping"
        case .unknown(let t): t
        }
    }
}

public struct CoreFrame: Sendable, Equatable, Codable {
    public var id: String?
    public var replyTo: String?
    public var event: CoreEvent

    public init(id: String? = nil, replyTo: String? = nil, event: CoreEvent) {
        self.id = id
        self.replyTo = replyTo
        self.event = event
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: FrameKeys.self)
        let v = try c.decode(Int.self, forKey: .v)
        guard v == GlideWire.version else { throw ProtocolError.unsupportedVersion(v) }
        let type = try c.decode(String.self, forKey: .type)
        id = try c.decodeIfPresent(String.self, forKey: .id)
        replyTo = try c.decodeIfPresent(String.self, forKey: .replyTo)
        switch type {
        case "hello": event = .hello(try c.decode(CoreHello.self, forKey: .data))
        case "state": event = .state(try c.decode(StateEvent.self, forKey: .data))
        case "transcript": event = .transcript(try c.decode(TranscriptEvent.self, forKey: .data))
        case "speech": event = .speech(try c.decode(SpeechEvent.self, forKey: .data))
        case "level": event = .level(try c.decode(LevelEvent.self, forKey: .data))
        case "task": event = .task(try c.decode(TaskEvent.self, forKey: .data))
        case "switch": event = .switchNotice(try c.decode(SwitchNotice.self, forKey: .data))
        case "approval_request": event = .approvalRequest(try c.decode(ApprovalRequest.self, forKey: .data))
        case "settings": event = .settings(try c.decode(SettingsReply.self, forKey: .data))
        case "settings_result": event = .settingsResult(try c.decode(SettingsResult.self, forKey: .data))
        case "error": event = .error(try c.decode(ErrorEvent.self, forKey: .data))
        case "ping": event = .ping
        default: event = .unknown(type: type)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: FrameKeys.self)
        try c.encode(GlideWire.version, forKey: .v)
        try c.encode(event.typeName, forKey: .type)
        try c.encodeIfPresent(id, forKey: .id)
        try c.encodeIfPresent(replyTo, forKey: .replyTo)
        switch event {
        case .hello(let p): try c.encode(p, forKey: .data)
        case .state(let p): try c.encode(p, forKey: .data)
        case .transcript(let p): try c.encode(p, forKey: .data)
        case .speech(let p): try c.encode(p, forKey: .data)
        case .level(let p): try c.encode(p, forKey: .data)
        case .task(let p): try c.encode(p, forKey: .data)
        case .switchNotice(let p): try c.encode(p, forKey: .data)
        case .approvalRequest(let p): try c.encode(p, forKey: .data)
        case .settings(let p): try c.encode(p, forKey: .data)
        case .settingsResult(let p): try c.encode(p, forKey: .data)
        case .error(let p): try c.encode(p, forKey: .data)
        case .ping, .unknown: break
        }
    }
}

// MARK: App to core

public enum AppCommand: Sendable, Equatable {
    case hello(AppHello)
    case textInput(TextInput)
    /// Cut off what the assistant is saying (barge-in). Does not stop a running task.
    case interrupt
    /// Stop the current task, or the named one, and any speech.
    case stop(StopRequest)
    case approvalResponse(ApprovalResponse)
    case voiceControl(VoiceControl)
    case settingsGet
    case settingsSet(SettingsSet)
    case pong

    public var typeName: String {
        switch self {
        case .hello: "hello"
        case .textInput: "text_input"
        case .interrupt: "interrupt"
        case .stop: "stop"
        case .approvalResponse: "approval_response"
        case .voiceControl: "voice_control"
        case .settingsGet: "settings_get"
        case .settingsSet: "settings_set"
        case .pong: "pong"
        }
    }
}

public struct AppFrame: Sendable, Equatable, Codable {
    public var id: String?
    public var command: AppCommand

    public init(id: String? = nil, command: AppCommand) {
        self.id = id
        self.command = command
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: FrameKeys.self)
        let v = try c.decode(Int.self, forKey: .v)
        guard v == GlideWire.version else { throw ProtocolError.unsupportedVersion(v) }
        let type = try c.decode(String.self, forKey: .type)
        id = try c.decodeIfPresent(String.self, forKey: .id)
        switch type {
        case "hello": command = .hello(try c.decode(AppHello.self, forKey: .data))
        case "text_input": command = .textInput(try c.decode(TextInput.self, forKey: .data))
        case "interrupt": command = .interrupt
        case "stop": command = .stop(try c.decodeIfPresent(StopRequest.self, forKey: .data) ?? StopRequest())
        case "approval_response": command = .approvalResponse(try c.decode(ApprovalResponse.self, forKey: .data))
        case "voice_control": command = .voiceControl(try c.decode(VoiceControl.self, forKey: .data))
        case "settings_get": command = .settingsGet
        case "settings_set": command = .settingsSet(try c.decode(SettingsSet.self, forKey: .data))
        case "pong": command = .pong
        default: throw ProtocolError.malformed("unknown command type")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: FrameKeys.self)
        try c.encode(GlideWire.version, forKey: .v)
        try c.encode(command.typeName, forKey: .type)
        try c.encodeIfPresent(id, forKey: .id)
        switch command {
        case .hello(let p): try c.encode(p, forKey: .data)
        case .textInput(let p): try c.encode(p, forKey: .data)
        case .stop(let p): try c.encode(p, forKey: .data)
        case .approvalResponse(let p): try c.encode(p, forKey: .data)
        case .voiceControl(let p): try c.encode(p, forKey: .data)
        case .settingsSet(let p): try c.encode(p, forKey: .data)
        case .interrupt, .settingsGet, .pong: break
        }
    }
}
