import Foundation
import GlideClient
import GlideProtocol

/// The pet's moods. They mirror the raccoon in the earlier PySide6 pet (idle, listening, typing, thinking,
/// talking, question, happy, sad, sleeping) so the behaviour carries over; the drawing here is placeholder shapes.
public enum PetState: String, CaseIterable, Sendable {
    case idle, listening, typing, thinking, talking, question, happy, sad, sleeping
}

public enum PetEyes: Sendable { case open, wide, closed, happy, sad }
public enum PetMouth: Sendable { case none, smile, open, frown }
public enum PetProp: Sendable { case none, sound, sound2, question, spark, sweat, z }

/// One frame of an animation. Plain data so it can be tested without a view.
public struct PetPose: Equatable, Sendable {
    public var dy = 0          // body offset in logical pixels, negative is up
    public var look = 0        // -1 left, 0 centre, 1 right
    public var tail = 0        // -1, 0, 1 sway
    public var eyes = PetEyes.open
    public var mouth = PetMouth.none
    public var armsUp = false
    public var laptop = false
    public var perk = false    // ears up, attentive
    public var prop = PetProp.none

    public init(dy: Int = 0, look: Int = 0, tail: Int = 0, eyes: PetEyes = .open, mouth: PetMouth = .none,
                armsUp: Bool = false, laptop: Bool = false, perk: Bool = false, prop: PetProp = .none) {
        self.dy = dy; self.look = look; self.tail = tail; self.eyes = eyes; self.mouth = mouth
        self.armsUp = armsUp; self.laptop = laptop; self.perk = perk; self.prop = prop
    }
}

extension PetState {
    public var frames: [PetPose] {
        switch self {
        case .idle:
            [PetPose(), PetPose(), PetPose(dy: 1, tail: 1), PetPose(dy: 1, tail: 1, eyes: .closed), PetPose(), PetPose(tail: -1)]
        case .listening:
            [(0, true), (0, false), (-1, true), (-1, false), (1, true), (1, false)]
                .map { PetPose(look: $0.0, eyes: $0.1 ? .wide : .open, perk: true) }
        case .typing:
            [(0, 0), (0, 1), (1, 1), (1, 0), (0, -1), (0, 0)].map { PetPose(look: $0.0, tail: $0.1, perk: true) }
        case .thinking:
            (0..<6).map { PetPose(look: [-1, -1, 0, 1, 1, 0][$0], tail: ($0 % 3) - 1, laptop: true) }
        case .talking:
            [PetPose(mouth: .open, prop: .sound), PetPose(eyes: .happy, mouth: .smile, prop: .sound2),
             PetPose(tail: 1, mouth: .open, prop: .sound), PetPose(tail: 1, prop: .sound2)]
        case .question:
            [(1, PetEyes.open, 0), (1, .open, 1), (1, .wide, 1), (0, .open, 0), (-1, .open, -1), (0, .closed, 0)]
                .map { PetPose(look: $0.0, tail: $0.2, eyes: $0.1, prop: .question) }
        case .happy:
            [PetPose(dy: 1, eyes: .happy), PetPose(dy: -2, eyes: .happy, mouth: .open, armsUp: true),
             PetPose(dy: -4, tail: 1, eyes: .happy, mouth: .open, armsUp: true, prop: .spark),
             PetPose(dy: -2, tail: -1, eyes: .happy, mouth: .smile, armsUp: true), PetPose(eyes: .happy, mouth: .smile)]
        case .sad:
            [PetProp.none, .sweat, .sweat, .sweat, .none, .none].map { PetPose(dy: 1, eyes: .sad, mouth: .frown, prop: $0) }
        case .sleeping:
            [PetProp.none, .none, .z, .z, .z, .none].map { PetPose(dy: 1, eyes: .closed, prop: $0) }
        }
    }

    /// Seconds each frame is held.
    public var frameInterval: TimeInterval {
        switch self {
        case .idle: 0.26
        case .listening: 0.20
        case .typing: 0.24
        case .thinking: 0.15
        case .talking: 0.15
        case .question: 0.22
        case .happy: 0.13
        case .sad: 0.20
        case .sleeping: 0.42
        }
    }

    public func pose(at time: TimeInterval) -> PetPose {
        let f = frames
        return f[Int(max(0, time) / frameInterval) % f.count]
    }
}

/// A finished task colours the pet for a few seconds.
public struct PetOutcome: Equatable, Sendable {
    public var state: PetState  // .happy or .sad
    public var until: Date
    public init(state: PetState, until: Date) {
        self.state = state
        self.until = until
    }
}

public struct PetInputs: Sendable {
    public var connected: Bool
    public var assistant: AssistantState
    public var muted: Bool
    public var userIsTyping: Bool
    public var approvalPending: Bool
    public var needsReconcile: Bool
    public var outcome: PetOutcome?

    public init(connected: Bool, assistant: AssistantState, muted: Bool = false, userIsTyping: Bool = false,
                approvalPending: Bool = false, needsReconcile: Bool = false, outcome: PetOutcome? = nil) {
        self.connected = connected; self.assistant = assistant; self.muted = muted; self.userIsTyping = userIsTyping
        self.approvalPending = approvalPending; self.needsReconcile = needsReconcile; self.outcome = outcome
    }
}

extension PetState {
    /// The mood for what the app knows right now. A disconnected core is sleeping, and anything that needs the
    /// user's attention (an approval, an unknown outcome) wins over an animation.
    public static func derive(_ i: PetInputs, now: Date) -> PetState {
        guard i.connected else { return .sleeping }
        if i.approvalPending || i.needsReconcile || i.assistant == .asking || i.assistant == .awaitingApproval { return .question }
        if let o = i.outcome, now < o.until { return o.state }
        switch i.assistant {
        case .error: return .sad
        case .speaking: return .talking
        case .thinking, .acting: return .thinking
        case .listening: return i.muted ? .idle : .listening
        case .idle, .unknown, .asking, .awaitingApproval: return i.userIsTyping ? .typing : .idle
        }
    }
}
