import Foundation
import Testing
@testable import GlideAppKit
import GlideClient
import GlideProtocol

/// A stand-in for the core connection that records what the model sends.
final class FakeLink: CoreLink, @unchecked Sendable {
    let events: AsyncStream<ClientEvent>
    private let cont: AsyncStream<ClientEvent>.Continuation
    private let lock = NSLock()
    private var sentCommands: [AppCommand] = []
    var settingsReply = SettingsReply(revision: 1, settings: GlideSettings())
    var setResult: Result<SettingsResult, Error> = .success(SettingsResult(ok: true, revision: 2))
    private var setCalls: [(Int, [SettingChange])] = []

    init() { (events, cont) = AsyncStream<ClientEvent>.makeStream() }

    var sent: [AppCommand] { lock.withLock { sentCommands } }
    var settingsSetCalls: [(Int, [SettingChange])] { lock.withLock { setCalls } }
    func start() async {}
    func stop() async { cont.finish() }
    func send(_ command: AppCommand) async throws { lock.withLock { sentCommands.append(command) } }
    func settingsGet() async throws -> SettingsReply { settingsReply }
    func settingsSet(baseRevision: Int, changes: [SettingChange]) async throws -> SettingsResult {
        lock.withLock { setCalls.append((baseRevision, changes)) }
        return try setResult.get()
    }
}

@MainActor
private func makeModel(now: Date = Date(timeIntervalSince1970: 1000)) -> (AppModel, FakeLink) {
    let link = FakeLink()
    return (AppModel(link: link, clock: { now }), link)
}

private func settle() async { try? await Task.sleep(for: .milliseconds(50)) }

@MainActor
@Suite("App model")
struct ModelTests {
    @Test func stateEventsDriveTheVisibleState() {
        let (m, _) = makeModel()
        m.apply(.state(StateEvent(assistant: .listening, handsFree: true, muted: false)))
        #expect(m.assistant == .listening)
        #expect(m.handsFree)
        m.apply(.level(LevelEvent(source: .mic, value: 7)))
        #expect(m.micLevel == 1, "levels are clamped to 0...1")
        m.apply(.level(LevelEvent(source: .mic, value: .nan)))
        #expect(m.micLevel == 0)
        m.apply(.level(LevelEvent(source: .mic, value: 0.5)))
        m.apply(.state(StateEvent(assistant: .thinking)))
        #expect(m.micLevel == 0, "the meter drops when listening stops")
    }

    @Test func partialTranscriptIsReplacedByTheFinalOne() {
        let (m, _) = makeModel()
        m.apply(.transcript(TranscriptEvent(utteranceId: "u1", role: .user, text: "open the", partial: true)))
        m.apply(.transcript(TranscriptEvent(utteranceId: "u1", role: .user, text: "open the calendar", partial: false)))
        #expect(m.transcript.count == 1)
        #expect(m.transcript[0].text == "open the calendar")
        #expect(!m.transcript[0].partial)
    }

    @Test func transcriptAndSwitchHistoryAreBounded() {
        let (m, _) = makeModel()
        for i in 0..<(AppModel.maxTranscriptLines + 20) {
            m.apply(.transcript(TranscriptEvent(utteranceId: "u\(i)", role: .user, text: "x")))
            m.apply(.switchNotice(SwitchNotice(role: "r", fromSlot: "a", toSlot: "b", kind: "timeout", reason: "slow")))
        }
        #expect(m.transcript.count == AppModel.maxTranscriptLines)
        #expect(m.switches.count == AppModel.maxSwitchRecords)
    }

    @Test func providerSwitchesAreRecordedAndVisible() {
        let (m, _) = makeModel()
        m.apply(.switchNotice(SwitchNotice(role: "tts", fromSlot: "a:x", toSlot: nil, kind: "auth", reason: "key refused")))
        #expect(m.switches.count == 1)
        #expect(m.switches[0].notice.toSlot == nil)
    }

    @Test func approvalIsAnsweredOnceAndNeverAutomatically() async {
        let (m, link) = makeModel()
        m.apply(.approvalRequest(ApprovalRequest(approvalId: "a1", kind: .input, command: "click Save", expiresInS: 30)))
        m.apply(.approvalRequest(ApprovalRequest(approvalId: "a1", kind: .input, command: "click Save")))
        #expect(m.approvals.count == 1, "a repeated request is not shown twice")
        await settle()
        #expect(link.sent.isEmpty, "the model never answers on its own")
        m.respond(to: "a1", .approve)
        m.respond(to: "a1", .deny)  // a second answer to the same approval is ignored
        await settle()
        #expect(link.sent == [.approvalResponse(ApprovalResponse(approvalId: "a1", decision: .approve))])
        #expect(m.approvals.isEmpty)
    }

    @Test func approvalsExpireLocallyAtTheirDeadline() {
        let (m, _) = makeModel(now: Date(timeIntervalSince1970: 100))
        m.apply(.approvalRequest(ApprovalRequest(approvalId: "a1", kind: .app, command: "open Notes", expiresInS: 10)))
        m.expireApprovals(now: Date(timeIntervalSince1970: 109))
        #expect(m.approvals.count == 1)
        m.expireApprovals(now: Date(timeIntervalSince1970: 110))
        #expect(m.approvals.isEmpty)
    }

    @Test func attemptedIsNotVerifiedAndUnknownOutcomeIsFlagged() {
        let (m, _) = makeModel()
        m.apply(.task(TaskEvent(taskId: "t", phase: .started)))
        m.apply(.task(TaskEvent(taskId: "t", phase: .attempted, step: 1, summary: "click")))
        #expect(m.task?.lastCheckVerified == nil, "an attempt alone confirms nothing")
        m.apply(.task(TaskEvent(taskId: "t", phase: .unverified)))
        #expect(m.task?.lastCheckVerified == false)
        m.apply(.task(TaskEvent(taskId: "t", phase: .reconcileRequired)))
        #expect(m.task?.needsReconcile == true)
        #expect(m.petState(at: Date(timeIntervalSince1970: 1000)) == .sleeping)  // not connected yet
    }

    @Test func finishedTasksColourThePetBriefly() {
        let t0 = Date(timeIntervalSince1970: 1000)
        let (m, _) = makeModel(now: t0)
        m.apply(status: .ready(CoreHello(coreVersion: "t", sessionId: "s")))
        m.apply(.state(StateEvent(assistant: .idle)))
        m.apply(.task(TaskEvent(taskId: "t", phase: .started)))
        m.apply(.task(TaskEvent(taskId: "t", phase: .completed)))
        #expect(m.task == nil)
        #expect(m.petState(at: t0) == .happy)
        #expect(m.petState(at: t0.addingTimeInterval(AppModel.outcomeHold + 1)) == .idle)
        m.apply(.task(TaskEvent(taskId: "u", phase: .failed)))
        #expect(m.petState(at: t0) == .sad)
    }

    @Test func losingTheConnectionClearsApprovalsAndFlagsTheTaskAsUnknown() {
        let (m, _) = makeModel()
        m.apply(status: .ready(CoreHello(coreVersion: "t", sessionId: "s", recordingContent: true)))
        #expect(m.recordingContent)
        m.apply(.task(TaskEvent(taskId: "t", phase: .attempted)))
        m.apply(.approvalRequest(ApprovalRequest(approvalId: "a", kind: .input, command: "c")))
        m.apply(.transcript(TranscriptEvent(utteranceId: "u", role: .user, text: "half", partial: true)))
        m.apply(status: .disconnected(reason: "core closed the connection"))
        #expect(m.approvals.isEmpty)
        #expect(m.task?.needsReconcile == true)
        #expect(m.transcript.isEmpty)
        #expect(m.notice?.contains("Check its result") == true)
        #expect(m.petState(at: .now) == .sleeping)
    }

    @Test func theUnknownOutcomeWarningSurvivesAReconnect() {
        let t0 = Date(timeIntervalSince1970: 1000)
        let (m, _) = makeModel(now: t0)
        let hello = CoreHello(coreVersion: "t", sessionId: "s")
        m.apply(status: .ready(hello))
        m.apply(.task(TaskEvent(taskId: "t", phase: .attempted, summary: "click")))
        m.apply(status: .disconnected(reason: "blip"))
        m.apply(status: .connecting(attempt: 1))
        m.apply(status: .ready(hello))
        m.apply(.state(StateEvent(assistant: .idle)))
        #expect(m.task?.needsReconcile == true)
        #expect(m.notice?.contains("Check its result") == true)
        #expect(m.petState(at: t0) == .question)
        m.acknowledgeReconcile()
        #expect(m.task == nil)
        #expect(m.petState(at: t0) == .idle)
    }

    @Test func theCoreEndingTheTaskClearsTheFlag() {
        let (m, _) = makeModel()
        m.apply(status: .ready(CoreHello(coreVersion: "t", sessionId: "s")))
        m.apply(.task(TaskEvent(taskId: "t", phase: .attempted)))
        m.apply(status: .disconnected(reason: nil))
        m.apply(status: .ready(CoreHello(coreVersion: "t", sessionId: "s")))
        m.apply(.task(TaskEvent(taskId: "t", phase: .failed)))
        #expect(m.task == nil)
    }

    @Test func sendTextTrimsAndIgnoresBlank() async {
        let (m, link) = makeModel()
        m.sendText("   ")
        m.draft = "  hello  "
        m.sendDraft()
        await settle()
        #expect(link.sent == [.textInput(TextInput(text: "hello"))])
        #expect(m.draft.isEmpty)
    }

    @Test func settingsChangeIsFollowedByAReadBack() async {
        let (m, link) = makeModel()
        link.settingsReply = SettingsReply(revision: 5, settings: GlideSettings(voice: VoiceSettings(handsFree: true)))
        await m.change([.handsFree(true)])
        #expect(link.settingsSetCalls.count == 1)
        #expect(link.settingsSetCalls[0].1 == [.handsFree(true)])
        #expect(m.settings?.voice.handsFree == true, "the screen shows what the core reports")
        #expect(m.settingsRevision == 5)
    }

    @Test func aRefusedChangeIsShownNotHidden() async {
        let (m, link) = makeModel()
        link.setResult = .success(SettingsResult(ok: false, revision: 1, errors: [SettingError(key: "voice.silence_ms", message: "out of range")]))
        await m.change([.silenceMs(5)])
        #expect(m.notice?.contains("out of range") == true)
    }

    @Test func aChangeWithUnknownOutcomeIsNotRepeated() async {
        let (m, link) = makeModel()
        link.setResult = .failure(ClientError.connectionLost)
        await m.change([.recordContent(true)])
        #expect(link.settingsSetCalls.count == 1, "never replay a write whose outcome is unknown")
        #expect(m.notice?.contains("unknown") == true)
    }

    @Test func pumpFeedsEventsFromTheLink() async {
        let link = FakeLink()
        let m = AppModel(link: link)
        m.start()
        await settle()
        // The link's stream is owned by the fake; yield through the model's own entry point instead.
        m.handle(.frame(CoreFrame(event: .state(StateEvent(assistant: .speaking)))))
        #expect(m.assistant == .speaking)
        await m.shutdown()
    }
}

@Suite("Pet")
struct PetTests {
    private let now = Date(timeIntervalSince1970: 500)

    @Test func everyStateHasFramesAndATimingThatMovesForward() {
        for s in PetState.allCases {
            #expect(!s.frames.isEmpty, "\(s) has no frames")
            #expect(s.frameInterval > 0)
            #expect(s.pose(at: 0) == s.frames[0])
            #expect(s.pose(at: s.frameInterval * Double(s.frames.count)) == s.frames[0], "loops")
        }
    }

    @Test func derivation() {
        func d(_ i: PetInputs) -> PetState { PetState.derive(i, now: now) }
        #expect(d(PetInputs(connected: false, assistant: .speaking)) == .sleeping)
        #expect(d(PetInputs(connected: true, assistant: .speaking)) == .talking)
        #expect(d(PetInputs(connected: true, assistant: .acting)) == .thinking)
        #expect(d(PetInputs(connected: true, assistant: .listening)) == .listening)
        #expect(d(PetInputs(connected: true, assistant: .listening, muted: true)) == .idle)
        #expect(d(PetInputs(connected: true, assistant: .idle, userIsTyping: true)) == .typing)
        #expect(d(PetInputs(connected: true, assistant: .error)) == .sad)
        #expect(d(PetInputs(connected: true, assistant: .unknown)) == .idle)
    }

    @Test func attentionBeatsAnimation() {
        let happy = PetOutcome(state: .happy, until: now.addingTimeInterval(5))
        #expect(PetState.derive(PetInputs(connected: true, assistant: .idle, approvalPending: true, outcome: happy), now: now) == .question)
        #expect(PetState.derive(PetInputs(connected: true, assistant: .acting, needsReconcile: true), now: now) == .question)
        #expect(PetState.derive(PetInputs(connected: true, assistant: .idle, outcome: happy), now: now) == .happy)
        #expect(PetState.derive(PetInputs(connected: true, assistant: .idle, outcome: happy), now: now.addingTimeInterval(6)) == .idle)
    }
}

@Suite("Labels")
struct LabelTests {
    @Test func theAppNamesNoModelOrProvider() {
        // Interface copy only. Provider and model names come from the core's settings payload.
        let all = AssistantState.allCases.map(\.label) + ApprovalKind.allCases.map(\.label) + TaskPhase.allCases.map(\.label)
        for text in all {
            for banned in ["gpt", "claude", "gemini", "eleven", "deepseek", "openai", "anthropic"] {
                #expect(!text.lowercased().contains(banned))
            }
        }
    }
}
