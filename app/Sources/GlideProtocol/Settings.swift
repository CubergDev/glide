import Foundation

// The settings payload. Provider, model, voice and role names are whatever the core reports; the app has no
// list of its own. Credentials never appear: a slot reports the *name* of its key variable and whether it is set.

public struct VoiceSettings: Codable, Sendable, Equatable {
    public var handsFree: Bool
    public var headset: Bool
    public var silenceMs: Int
    public var language: String?
    /// The accepted range for `silenceMs`, from the core, so the app does not duplicate its limits.
    public var silenceMsRange: [Int]?

    public init(handsFree: Bool = false, headset: Bool = false, silenceMs: Int = 600, language: String? = nil,
                silenceMsRange: [Int]? = nil) {
        self.handsFree = handsFree
        self.headset = headset
        self.silenceMs = silenceMs
        self.language = language
        self.silenceMsRange = silenceMsRange
    }

    enum CodingKeys: String, CodingKey {
        case handsFree = "hands_free", headset, silenceMs = "silence_ms", language, silenceMsRange = "silence_ms_range"
    }
}

public struct PrivacySettings: Codable, Sendable, Equatable {
    /// Detailed recording of utterances, typed text and captured content. Off unless the user opts in.
    public var recordContent: Bool
    public init(recordContent: Bool = false) { self.recordContent = recordContent }
    enum CodingKeys: String, CodingKey { case recordContent = "record_content" }
}

public struct ComputerSettings: Codable, Sendable, Equatable {
    /// Whether the assistant may drive the screen at all. Each use still needs an approval.
    public var actEnabled: Bool
    public init(actEnabled: Bool = false) { self.actEnabled = actEnabled }
    enum CodingKeys: String, CodingKey { case actEnabled = "act_enabled" }
}

public struct SlotSettings: Codable, Sendable, Equatable, Identifiable {
    public var name: String
    public var provider: String
    public var model: String?
    /// The name of the environment variable that holds the key. Never the key.
    public var keyEnv: String?
    public var keyPresent: Bool
    public var status: SlotStatus
    public var id: String { name }

    public init(name: String, provider: String, model: String? = nil, keyEnv: String? = nil,
                keyPresent: Bool = false, status: SlotStatus = .unknown) {
        self.name = name
        self.provider = provider
        self.model = model
        self.keyEnv = keyEnv
        self.keyPresent = keyPresent
        self.status = status
    }

    enum CodingKeys: String, CodingKey {
        case name, provider, model, keyEnv = "key_env", keyPresent = "key_present", status
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = try c.decode(String.self, forKey: .name)
        provider = try c.decode(String.self, forKey: .provider)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        keyEnv = try c.decodeIfPresent(String.self, forKey: .keyEnv)
        keyPresent = try c.decodeIfPresent(Bool.self, forKey: .keyPresent) ?? false
        status = try c.decodeIfPresent(SlotStatus.self, forKey: .status) ?? .unknown
    }
}

public struct RoleSettings: Codable, Sendable, Equatable, Identifiable {
    /// The role as the core names it, for example the key of a chain table in `glide.toml`.
    public var role: String
    public var chain: [SlotSettings]
    /// The slot name this role is pinned to, if any.
    public var pinned: String?
    public var id: String { role }

    public init(role: String, chain: [SlotSettings], pinned: String? = nil) {
        self.role = role
        self.chain = chain
        self.pinned = pinned
    }
}

public struct GlideSettings: Codable, Sendable, Equatable {
    public var voice: VoiceSettings
    public var privacy: PrivacySettings
    public var computer: ComputerSettings
    public var roles: [RoleSettings]

    public init(voice: VoiceSettings = .init(), privacy: PrivacySettings = .init(),
                computer: ComputerSettings = .init(), roles: [RoleSettings] = []) {
        self.voice = voice
        self.privacy = privacy
        self.computer = computer
        self.roles = roles
    }
}

/// A reply to `settings_get`, or an unsolicited push after the core's settings changed.
public struct SettingsReply: Codable, Sendable, Equatable {
    public var revision: Int
    public var settings: GlideSettings
    public init(revision: Int, settings: GlideSettings) {
        self.revision = revision
        self.settings = settings
    }
}

public struct SettingError: Codable, Sendable, Equatable {
    public var key: String
    public var message: String
    public init(key: String, message: String) {
        self.key = key
        self.message = message
    }
}

/// A reply to `settings_set`. When `ok` is false nothing was applied.
public struct SettingsResult: Codable, Sendable, Equatable {
    public var ok: Bool
    public var revision: Int
    public var errors: [SettingError]
    public init(ok: Bool, revision: Int, errors: [SettingError] = []) {
        self.ok = ok
        self.revision = revision
        self.errors = errors
    }

    enum CodingKeys: String, CodingKey { case ok, revision, errors }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        ok = try c.decode(Bool.self, forKey: .ok)
        revision = try c.decode(Int.self, forKey: .revision)
        errors = try c.decodeIfPresent([SettingError].self, forKey: .errors) ?? []
    }
}

/// One change in a `settings_set`. A closed set of keys: the app cannot send an arbitrary key, so it cannot
/// send a credential through settings. The core validates ranges and may reject a change.
public enum SettingChange: Sendable, Equatable, Codable {
    case handsFree(Bool)
    case headset(Bool)
    case silenceMs(Int)
    case language(String)
    case recordContent(Bool)
    case actEnabled(Bool)
    /// Pin a role to one slot, or clear the pin with nil.
    case pin(role: String, slot: String?)

    enum CodingKeys: String, CodingKey { case key, value }
    enum PinKeys: String, CodingKey { case role, slot }

    public var key: String {
        switch self {
        case .handsFree: "voice.hands_free"
        case .headset: "voice.headset"
        case .silenceMs: "voice.silence_ms"
        case .language: "voice.language"
        case .recordContent: "privacy.record_content"
        case .actEnabled: "computer.act_enabled"
        case .pin: "roles.pin"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let key = try c.decode(String.self, forKey: .key)
        switch key {
        case "voice.hands_free": self = .handsFree(try c.decode(Bool.self, forKey: .value))
        case "voice.headset": self = .headset(try c.decode(Bool.self, forKey: .value))
        case "voice.silence_ms": self = .silenceMs(try c.decode(Int.self, forKey: .value))
        case "voice.language": self = .language(try c.decode(String.self, forKey: .value))
        case "privacy.record_content": self = .recordContent(try c.decode(Bool.self, forKey: .value))
        case "computer.act_enabled": self = .actEnabled(try c.decode(Bool.self, forKey: .value))
        case "roles.pin":
            let p = try c.nestedContainer(keyedBy: PinKeys.self, forKey: .value)
            self = .pin(role: try p.decode(String.self, forKey: .role), slot: try p.decodeIfPresent(String.self, forKey: .slot))
        default:
            throw DecodingError.dataCorruptedError(forKey: .key, in: c, debugDescription: "unknown setting key")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(key, forKey: .key)
        switch self {
        case .handsFree(let v), .headset(let v), .recordContent(let v), .actEnabled(let v): try c.encode(v, forKey: .value)
        case .silenceMs(let v): try c.encode(v, forKey: .value)
        case .language(let v): try c.encode(v, forKey: .value)
        case .pin(let role, let slot):
            var p = c.nestedContainer(keyedBy: PinKeys.self, forKey: .value)
            try p.encode(role, forKey: .role)
            try p.encode(slot, forKey: .slot)
        }
    }
}

public struct SettingsSet: Codable, Sendable, Equatable {
    /// The revision the app last saw. The core refuses a change made against a stale revision.
    public var baseRevision: Int
    public var changes: [SettingChange]
    public init(baseRevision: Int, changes: [SettingChange]) {
        self.baseRevision = baseRevision
        self.changes = changes
    }
    enum CodingKeys: String, CodingKey { case baseRevision = "base_revision", changes }
}
