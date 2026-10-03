// swift-tools-version: 6.0
import PackageDescription

// The native Mac app for Glide (decision D11). It is a separate program from the Python core and talks to it
// over a local Unix domain socket using the JSON-lines protocol in PROTOCOL.md.
//
//   GlideProtocol  wire models, codec and line framing. Pure Foundation, no sockets, no UI.
//   GlideClient    Unix-socket transport and the reconnecting client. No UI.
//   GlideAppKit    app state (AppModel), pet state machine, SwiftUI views. No @main.
//   GlideApp       the executable: MenuBarExtra and Settings scenes only.
let package = Package(
    name: "GlideApp",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "GlideApp", targets: ["GlideApp"]),
        .library(name: "GlideProtocol", targets: ["GlideProtocol"]),
        .library(name: "GlideClient", targets: ["GlideClient"]),
    ],
    targets: [
        .target(name: "GlideProtocol"),
        .target(name: "GlideClient", dependencies: ["GlideProtocol"]),
        .target(name: "GlideAppKit", dependencies: ["GlideProtocol", "GlideClient"]),
        .executableTarget(name: "GlideApp", dependencies: ["GlideAppKit", "GlideClient", "GlideProtocol"]),
        .testTarget(name: "GlideProtocolTests", dependencies: ["GlideProtocol"]),
        .testTarget(name: "GlideClientTests", dependencies: ["GlideClient", "GlideProtocol"]),
        .testTarget(name: "GlideAppKitTests", dependencies: ["GlideAppKit", "GlideClient", "GlideProtocol"]),
    ]
)
