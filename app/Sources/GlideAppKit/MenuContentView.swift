import GlideClient
import GlideProtocol
import SwiftUI

/// The window shown from the menu bar item.
public struct MenuContentView: View {
    // No @State here: the SwiftUI @State macro needs Xcode's macro plugins, and this package builds with the
    // Command Line Tools alone. The two bits of view state live in AppModel instead.
    @Bindable var model: AppModel

    public init(model: AppModel) { self.model = model }

    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            if let notice = model.notice {
                HStack(alignment: .top) {
                    Text(verbatim: notice).font(.caption).frame(maxWidth: .infinity, alignment: .leading)
                    Button("Dismiss") { model.dismissNotice() }.controlSize(.mini)
                }
                .padding(8).background(.yellow.opacity(0.2), in: RoundedRectangle(cornerRadius: 6))
            }
            ForEach(model.approvals) { ApprovalCard(model: model, approval: $0) }
            if let task = model.task { TaskRow(task: task, onChecked: { model.acknowledgeReconcile() }) }
            VoiceStatusView(model: model)
            conversation
            composer
            controls
            if !model.switches.isEmpty { switchesSection }
            Divider()
            footer
        }
        .padding(14)
        .frame(width: 340)
    }

    private var header: some View {
        HStack(spacing: 12) {
            TimelineView(.periodic(from: .now, by: 0.5)) { t in
                PetView(state: model.petState(at: t.date)).frame(width: 64, height: 64)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(verbatim: model.isConnected ? model.assistant.label : "Not connected").font(.headline)
                Text(verbatim: model.connection.label).font(.caption).foregroundStyle(.secondary)
                if model.recordingContent {
                    Label("Recording content", systemImage: "record.circle").font(.caption).foregroundStyle(.red)
                }
            }
            Spacer()
        }
    }

    private var conversation: some View {
        let lines = model.transcript.suffix(4)
        return VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(lines)) { line in
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(verbatim: line.role == .user ? "You" : "Glide").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    Text(verbatim: line.redacted || line.text == nil ? "(hidden)" : line.text ?? "")
                        .font(.callout).opacity(line.partial ? 0.6 : 1)
                }
            }
        }
    }

    private var composer: some View {
        HStack {
            TextField("Type to Glide", text: $model.draft)
                .textFieldStyle(.roundedBorder)
                .onSubmit(send)
                                .disabled(!model.isConnected)
            Button("Send", action: send).disabled(!model.isConnected || model.draft.trimmingCharacters(in: .whitespaces).isEmpty)
        }
    }

    private func send() {
        model.sendDraft()
    }

    private var controls: some View {
        HStack {
            Button { model.interrupt() } label: { Label("Interrupt", systemImage: "speaker.slash") }
                .disabled(!model.isConnected || model.assistant != .speaking)
            Button(role: .destructive) { model.stop() } label: { Label("Stop", systemImage: "stop.circle") }
                .disabled(!model.isConnected)
            Spacer()
            Toggle("Hands-free", isOn: Binding(
                get: { model.handsFree },
                set: { on in Task { await model.change([.handsFree(on)]) } }))
                .toggleStyle(.switch).controlSize(.small)
                .disabled(!model.isConnected || model.settingsBusy)
        }
        .controlSize(.small)
    }

    private var switchesSection: some View {
        DisclosureGroup(isExpanded: $model.showSwitches) {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(model.switches.suffix(5).reversed()) { SwitchRow(record: $0) }
            }
        } label: {
            Label("Provider switches (\(model.switches.count))", systemImage: "arrow.triangle.swap").font(.caption)
        }
    }

    private var footer: some View {
        HStack {
            SettingsLink { Text("Settings") }
            Spacer()
            Button("Quit Glide") { NSApplication.shared.terminate(nil) }
        }
        .controlSize(.small)
    }
}

struct TaskRow: View {
    var task: ActiveTask
    var onChecked: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                ProgressView().controlSize(.small).opacity(task.needsReconcile ? 0 : 1)
                Text(verbatim: task.phase.label).font(.callout.weight(.medium))
                if let step = task.step { Text(verbatim: "step \(step)").font(.caption).foregroundStyle(.secondary) }
            }
            if let summary = task.summary { Text(verbatim: summary).font(.caption).foregroundStyle(.secondary).lineLimit(2) }
            if task.needsReconcile {
                Text("An action may or may not have taken effect. Glide will not repeat it. Check your screen, then tell Glide what to do.")
                    .font(.caption).foregroundStyle(.orange)
                Button("I checked", action: onChecked).controlSize(.small)
            } else if task.lastCheckVerified == false {
                Text("The last action could not be confirmed.").font(.caption).foregroundStyle(.orange)
            }
        }
    }
}

struct SwitchRow: View {
    var record: SwitchRecord

    var body: some View {
        let n = record.notice
        VStack(alignment: .leading, spacing: 1) {
            Text(verbatim: "\(n.role): \(n.fromSlot) to \(n.toSlot ?? "nothing left to try")").font(.caption.weight(.medium))
            Text(verbatim: "\(n.kind): \(n.reason)").font(.caption2).foregroundStyle(.secondary).lineLimit(2)
        }
    }
}
