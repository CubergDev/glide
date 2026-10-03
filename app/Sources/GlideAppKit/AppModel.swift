import Foundation
import GlideClient
import GlideProtocol
import Observation

public struct TranscriptLine: Identifiable, Equatable, Sendable {
    public var id: String  // utterance id plus role
    public var role: TranscriptRole
    public var text: String?
    public var partial: Bool
    public var redacted: Bool
}

public struct SwitchRecord: Identifiable, Equatable, Sendable {
    public let id: Int
    public let notice: SwitchNotice
    public let at: Date
}

public struct PendingApproval: Identifiable, Equatable, Sendable {
    public var request: ApprovalRequest
    public var receivedAt: Date
    public var deadline: Date?
    public var id: String { request.approvalId }
}

public struct ActiveTask: Equatable, Sendable {
    public var id: String
    public var phase: TaskPhase
    public var step: Int?
    public var summary: String?
    public var lastCheckVerified: Bool?
    /// A write has an unknown outcome. Nothing is replayed; the user checks and decides.
    public var needsReconcile: Bool
}

/// Everything the views show. Fed by `ClientEvent`s; sends commands through a `CoreLink`.
///
/// Transcript text and task summaries are held in memory only, bounded, and never written to disk.
@MainActor
@Observable
public final class AppModel {
    public static let maxTranscriptLines = 50
    public static let maxSwitchRecords = 50
    public static let outcomeHold: TimeInterval = 4

    public private(set) var connection: ConnectionStatus = .disconnected(reason: nil)
    public private(set) var assistant: AssistantState = .idle
    public private(set) var handsFree = false
    public private(set) var muted = false
    public private(set) var stateDetail: String?
    public private(set) var recordingContent = false
    public private(set) var micLevel = 0.0
    public private(set) var speakerLevel = 0.0
    public private(set) var transcript: [TranscriptLine] = []
    public private(set) var switches: [SwitchRecord] = []
    public private(set) var approvals: [PendingApproval] = []
    public private(set) var task: ActiveTask?
    public private(set) var outcome: PetOutcome?
    public private(set) var settings: GlideSettings?
    public private(set) var settingsRevision = 0
    public private(set) var settingsBusy = false
    public private(set) var notice: String?
    /// The text field's contents. Held here, not in the view, so the pet can show typing and the text is never
    /// stored anywhere but memory.
    public var draft = ""
    public var showSwitches = false
    public var userIsTyping: Bool { !draft.isEmpty }

    private let link: any CoreLink
    private let clock: @Sendable () -> Date
    private var pump: Task<Void, Never>?
    private var switchCounter = 0

    public init(link: any CoreLink, clock: @escaping @Sendable () -> Date = Date.init) {
        self.link = link
        self.clock = clock
    }

    // MARK: Lifecycle

    public func start() {
        guard pump == nil else { return }
        pump = Task { [link] in
            await link.start()
            for await event in link.events {
                self.handle(event)
            }
        }
    }

    public func shutdown() async {
        pump?.cancel()
        pump = nil
        await link.stop()
    }

    // MARK: Derived

    public var isConnected: Bool { connection.isReady }

    public func petState(at now: Date) -> PetState {
        PetState.derive(PetInputs(connected: isConnected, assistant: assistant, muted: muted, userIsTyping: userIsTyping,
                                  approvalPending: !approvals.isEmpty, needsReconcile: task?.needsReconcile ?? false,
                                  outcome: outcome), now: now)
    }

    // MARK: Events

    public func handle(_ event: ClientEvent) {
        switch event {
        case .status(let s): apply(status: s)
        case .frame(let f): apply(f.event)
        case .protocolIssue(let why): notice = "Ignored a message from the core: \(why)"
        }
    }

    func apply(status s: ConnectionStatus) {
        connection = s
        if case .ready(let hello) = s {
            recordingContent = hello.recordingContent
            // Keep the unknown-outcome warning across a reconnect: the app cannot learn how the task ended.
            if task?.needsReconcile != true { notice = nil }
            Task { await self.refreshSettings() }
        } else {
            // The core treats an unanswered approval as denied, and the app cannot know how a task ended.
            approvals = []
            assistant = .idle
            micLevel = 0
            speakerLevel = 0
            transcript.removeAll { $0.partial }
            if task != nil {
                // The app cannot know how the task ended. Flag it, as for a write with an unknown outcome, and
                // keep the flag until the core reports the task's end or the user says they checked.
                task?.needsReconcile = true
                notice = "Connection lost during a task. Check its result before repeating anything."
            }
        }
    }

    public func apply(_ event: CoreEvent) {
        let now = clock()
        switch event {
        case .hello, .ping, .unknown, .settingsResult: break
        case .state(let s):
            assistant = s.assistant
            handsFree = s.handsFree
            muted = s.muted
            stateDetail = s.detail
            if s.assistant != .listening { micLevel = 0 }
            if s.assistant != .speaking { speakerLevel = 0 }
        case .transcript(let t):
            let id = "\(t.utteranceId)/\(t.role.rawValue)"
            let line = TranscriptLine(id: id, role: t.role, text: t.text, partial: t.partial, redacted: t.redacted)
            if let i = transcript.firstIndex(where: { $0.id == id }) { transcript[i] = line } else { transcript.append(line) }
            if transcript.count > Self.maxTranscriptLines { transcript.removeFirst(transcript.count - Self.maxTranscriptLines) }
        case .speech(let s):
            if s.phase == .interrupted || s.phase == .finished { speakerLevel = 0 }
        case .level(let l):
            let v = min(1, max(0, l.value.isFinite ? l.value : 0))
            switch l.source {
            case .mic: micLevel = v
            case .speaker: speakerLevel = v
            case .unknown: break
            }
        case .task(let t): apply(task: t, now: now)
        case .switchNotice(let n):
            switchCounter += 1
            switches.append(SwitchRecord(id: switchCounter, notice: n, at: now))
            if switches.count > Self.maxSwitchRecords { switches.removeFirst(switches.count - Self.maxSwitchRecords) }
        case .approvalRequest(let r):
            guard !approvals.contains(where: { $0.id == r.approvalId }) else { break }
            let deadline = r.expiresInS.map { now.addingTimeInterval($0) }
            approvals.append(PendingApproval(request: r, receivedAt: now, deadline: deadline))
            if let seconds = r.expiresInS, seconds.isFinite, seconds >= 0 {
                // The core treats silence as a denial at the deadline, so the card goes away then too.
                Task { [clock] in
                    try? await Task.sleep(for: .seconds(seconds))
                    self.expireApprovals(now: clock())
                }
            }
        case .settings(let reply):
            settings = reply.settings
            settingsRevision = reply.revision
        case .error(let e):
            notice = "\(e.code): \(e.message)"
        }
    }

    private func apply(task t: TaskEvent, now: Date) {
        switch t.phase {
        case .completed:
            task = nil
            outcome = PetOutcome(state: .happy, until: now.addingTimeInterval(Self.outcomeHold))
        case .failed:
            task = nil
            outcome = PetOutcome(state: .sad, until: now.addingTimeInterval(Self.outcomeHold))
        case .stopped:
            task = nil
            outcome = nil
        case .started, .step, .attempted, .verified, .unverified, .reconcileRequired, .unknown:
            var cur = task?.id == t.taskId ? task! : ActiveTask(id: t.taskId, phase: t.phase, needsReconcile: false)
            cur.phase = t.phase
            if let s = t.step { cur.step = s }
            if let s = t.summary { cur.summary = s }
            if t.phase == .verified || t.phase == .unverified { cur.lastCheckVerified = t.verified ?? (t.phase == .verified) }
            if t.phase == .reconcileRequired { cur.needsReconcile = true }
            task = cur
        }
    }

    /// Drops approvals past their deadline. The core has already treated them as denied.
    public func expireApprovals(now: Date) {
        approvals.removeAll { a in a.deadline.map { now >= $0 } ?? false }
    }

    // MARK: Commands

    public func sendText(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        send(.textInput(TextInput(text: trimmed)), failure: "Not sent: the core is not connected.")
    }

    public func sendDraft() {
        let text = draft
        draft = ""
        sendText(text)
    }

    public func interrupt() { send(.interrupt, failure: "Could not interrupt: the core is not connected.") }

    public func stop() { send(.stop(StopRequest(taskId: task?.id)), failure: "Could not stop: the core is not connected.") }

    public func setMuted(_ on: Bool) {
        send(.voiceControl(VoiceControl(action: on ? .mute : .unmute)), failure: "Could not change the microphone: the core is not connected.")
    }

    /// Answers one approval, once. There is no "always".
    public func respond(to approvalId: String, _ decision: ApprovalDecision) {
        guard approvals.contains(where: { $0.id == approvalId }) else { return }
        approvals.removeAll { $0.id == approvalId }
        send(.approvalResponse(ApprovalResponse(approvalId: approvalId, decision: decision)),
             failure: "Your answer was not delivered. The core treats a missing answer as a denial.")
    }

    private func send(_ command: AppCommand, failure: String) {
        Task {
            do { try await link.send(command) } catch { self.notice = failure }
        }
    }

    // MARK: Settings

    public func refreshSettings() async {
        do {
            let reply = try await link.settingsGet()
            settings = reply.settings
            settingsRevision = reply.revision
        } catch {
            notice = "Could not read settings from the core."
        }
    }

    /// Sends a change and then re-reads the core's settings, so the screen shows what the core holds, not what the
    /// app hoped. A change whose outcome is unknown (connection lost, timeout) is never sent again by the app.
    public func change(_ changes: [SettingChange]) async {
        guard !settingsBusy else { return }
        settingsBusy = true
        defer { settingsBusy = false }
        do {
            let result = try await link.settingsSet(baseRevision: settingsRevision, changes: changes)
            if !result.ok {
                notice = result.errors.isEmpty ? "The core refused the change."
                    : "The core refused: " + result.errors.map { "\($0.key): \($0.message)" }.joined(separator: "; ")
            }
        } catch {
            notice = "The result of the change is unknown. Showing what the core has now."
        }
        await refreshSettings()
    }

    public func dismissNotice() { notice = nil }

    /// The user checked the screen and wants the unknown-outcome flag cleared.
    public func acknowledgeReconcile() {
        if task?.needsReconcile == true {
            task = nil
            notice = nil
        }
    }
}
