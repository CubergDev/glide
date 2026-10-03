import GlideClient
import GlideProtocol
import SwiftUI

/// Settings. Everything shown comes from the core's settings payload. Changes go to the core and the screen then
/// shows what the core reports, so a refused or lost change is visible instead of looking applied.
public struct SettingsView: View {
    var model: AppModel
    var socketPath: String

    public init(model: AppModel, socketPath: String) {
        self.model = model
        self.socketPath = socketPath
    }

    public var body: some View {
        TabView {
            general.tabItem { Label("General", systemImage: "gearshape") }
            voice.tabItem { Label("Voice", systemImage: "waveform") }
            providers.tabItem { Label("Providers", systemImage: "arrow.triangle.branch") }
            privacy.tabItem { Label("Privacy", systemImage: "lock") }
        }
        .frame(width: 520, height: 400)
        .padding()
    }

    private var general: some View {
        Form {
            LabeledContent("Core") { Text(verbatim: model.connection.label) }
            LabeledContent("Socket") { Text(verbatim: socketPath).textSelection(.enabled) }
            Toggle("Allow Glide to use the screen, mouse and keyboard", isOn: bind(\.computer.actEnabled, { .actEnabled($0) }))
            Text("Each use still asks you first, for that exact action.").font(.caption).foregroundStyle(.secondary)
            if let n = model.notice { Text(verbatim: n).font(.caption).foregroundStyle(.orange) }
            Button("Reload from core") { Task { await model.refreshSettings() } }.disabled(!model.isConnected)
        }
        .formStyle(.grouped)
        .disabled(model.settings == nil && model.isConnected)
    }

    private var voice: some View {
        Form {
            Toggle("Hands-free", isOn: bind(\.voice.handsFree, { .handsFree($0) }))
            Toggle("Headset mode", isOn: bind(\.voice.headset, { .headset($0) }))
            if let v = model.settings?.voice {
                let range = silenceRange(v)
                Stepper(value: Binding(get: { v.silenceMs }, set: { new in Task { await model.change([.silenceMs(new)]) } }),
                        in: range, step: 50) {
                    LabeledContent("Pause that ends a sentence") { Text(verbatim: "\(v.silenceMs) ms") }
                }
                if let lang = v.language { LabeledContent("Language") { Text(verbatim: lang) } }
            } else {
                Text("Connect to the core to see voice settings.").foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }

    private func silenceRange(_ v: VoiceSettings) -> ClosedRange<Int> {
        if let r = v.silenceMsRange, r.count == 2, r[0] <= r[1] { return r[0]...r[1] }
        return v.silenceMs...v.silenceMs  // the core did not say what is allowed, so do not invent a range
    }

    private var providers: some View {
        List {
            if let roles = model.settings?.roles, !roles.isEmpty {
                ForEach(roles) { role in
                    Section {
                        ForEach(role.chain) { slot in SlotRow(slot: slot, pinned: role.pinned == slot.name) }
                        Picker("Pin", selection: Binding(
                            get: { role.pinned ?? "" },
                            set: { new in Task { await model.change([.pin(role: role.role, slot: new.isEmpty ? nil : new)]) } })) {
                            Text("None, use the chain in order").tag("")
                            ForEach(role.chain) { Text(verbatim: $0.name).tag($0.name) }
                        }
                    } header: { Text(verbatim: role.role) }
                }
            } else {
                Text("No provider information yet. It comes from the core.").foregroundStyle(.secondary)
            }
            Text("Providers, models and keys are set in the core's configuration file and environment. This list is read-only except for pinning.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var privacy: some View {
        Form {
            Toggle("Record utterances, typed text and captured content", isOn: bind(\.privacy.recordContent, { .recordContent($0) }))
            Text("Off by default. When on, the core stores what you say and type, and what it sees on screen, with its run data. Turn it off to stop storing more.")
                .font(.caption).foregroundStyle(.secondary)
            Text("This app keeps what it shows in memory only and never writes it to disk.").font(.caption).foregroundStyle(.secondary)
        }
        .formStyle(.grouped)
    }

    private func bind(_ path: KeyPath<GlideSettings, Bool>, _ make: @escaping (Bool) -> SettingChange) -> Binding<Bool> {
        Binding(get: { model.settings?[keyPath: path] ?? false },
                set: { new in Task { await model.change([make(new)]) } })
    }
}

struct SlotRow: View {
    var slot: SlotSettings
    var pinned: Bool

    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(verbatim: slot.name).font(.callout)
                if let env = slot.keyEnv {
                    Text(verbatim: slot.keyPresent ? "\(env) is set" : "\(env) is not set").font(.caption)
                        .foregroundStyle(slot.keyPresent ? Color.secondary : Color.orange)
                }
            }
            Spacer()
            if pinned { Image(systemName: "pin.fill").accessibilityLabel("Pinned") }
            Text(verbatim: slot.status.rawValue).font(.caption).foregroundStyle(.secondary)
        }
    }
}
