import AppKit
import GlideAppKit
import GlideClient
import SwiftUI

/// Keeps the app out of the Dock and the app switcher: it lives in the menu bar.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
    }
}

@main
struct GlideApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    private let model: AppModel
    private let socketPath: String

    init() {
        let path = SocketPath.resolve(arguments: CommandLine.arguments, environment: ProcessInfo.processInfo.environment)
        socketPath = path
        let model = AppModel(link: GlideClient(socketPath: path))
        self.model = model
        model.start()
    }

    var body: some Scene {
        MenuBarExtra {
            MenuContentView(model: model)
        } label: {
            MenuBarLabel(model: model)
        }
        .menuBarExtraStyle(.window)

        Settings {
            SettingsView(model: model, socketPath: socketPath)
        }
    }
}

/// The menu bar icon. A mark appears when something needs the user.
struct MenuBarLabel: View {
    var model: AppModel

    var body: some View {
        Image(systemName: symbol)
            .accessibilityLabel(Text(verbatim: label))
    }

    private var symbol: String {
        if !model.isConnected { return "pawprint" }
        if !model.approvals.isEmpty || model.task?.needsReconcile == true { return "exclamationmark.bubble.fill" }
        if model.muted { return "mic.slash" }
        switch model.assistant {
        case .listening: return "waveform"
        case .speaking: return "speaker.wave.2.fill"
        case .thinking, .acting: return "ellipsis.circle"
        case .error: return "exclamationmark.triangle"
        default: return "pawprint.fill"
        }
    }

    private var label: String {
        if !model.approvals.isEmpty { return "Glide, waiting for your approval" }
        return model.isConnected ? "Glide, \(model.assistant.label)" : "Glide, not connected"
    }
}
