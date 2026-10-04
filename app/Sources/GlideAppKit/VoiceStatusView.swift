import GlideProtocol
import SwiftUI

/// Hands-free status: whether the mic is live, the level while listening, and the latest words.
public struct VoiceStatusView: View {
    var model: AppModel

    public init(model: AppModel) { self.model = model }

    public var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: icon).foregroundStyle(model.muted ? .secondary : Color.accentColor)
                Text(verbatim: model.muted ? "Microphone muted" : (model.handsFree ? "Hands-free on" : "Hands-free off"))
                    .font(.callout.weight(.medium))
                Spacer()
                Button(model.muted ? "Unmute" : "Mute") { model.setMuted(!model.muted) }
                    .controlSize(.small)
                    .disabled(!model.isConnected)
            }
            LevelBar(value: model.assistant == .speaking ? model.speakerLevel : model.micLevel,
                     active: !model.muted && (model.assistant == .listening || model.assistant == .speaking))
                .frame(height: 6)
            if let last = model.transcript.last(where: { $0.role == .user }) {
                Text(verbatim: last.redacted || last.text == nil ? "(hidden)" : last.text ?? "")
                    .font(.caption).foregroundStyle(last.partial ? .secondary : .primary)
                    .lineLimit(2)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var icon: String {
        if model.muted { return "mic.slash" }
        return model.assistant == .listening ? "waveform" : "mic"
    }
}

struct LevelBar: View {
    var value: Double
    var active: Bool

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(.quaternary)
                Capsule().fill(active ? Color.accentColor : .secondary)
                    .frame(width: max(4, geo.size.width * (active ? value : 0)))
                    .animation(.linear(duration: 0.08), value: value)
            }
        }
        .accessibilityHidden(true)
    }
}
