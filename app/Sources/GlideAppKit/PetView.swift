import SwiftUI

/// The raccoon as placeholder shapes. It animates from `PetState.frames`; replace the drawing later without
/// touching the state machine. With Reduce Motion on, it shows the first frame only.
public struct PetView: View {
    public var state: PetState
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    public init(state: PetState) { self.state = state }

    public var body: some View {
        TimelineView(.animation(minimumInterval: state.frameInterval, paused: reduceMotion)) { timeline in
            let pose = reduceMotion ? state.frames[0] : state.pose(at: timeline.date.timeIntervalSinceReferenceDate)
            Canvas { context, size in draw(pose, in: &context, size: size) }
                .aspectRatio(1, contentMode: .fit)
        }
        .accessibilityElement()
        .accessibilityLabel(Text(verbatim: "Raccoon, \(state.rawValue)"))
    }

    private func draw(_ pose: PetPose, in ctx: inout GraphicsContext, size: CGSize) {
        let u = min(size.width, size.height) / 32  // 32 logical units square
        func r(_ x: Double, _ y: Double, _ w: Double, _ h: Double) -> CGRect {
            CGRect(x: x * u, y: (y + Double(pose.dy) * 0.5) * u, width: w * u, height: h * u)
        }
        let fur = Color(white: 0.55), dark = Color(white: 0.18), light = Color(white: 0.93)

        // tail
        let sway = Double(pose.tail) * 1.5
        ctx.fill(Path(ellipseIn: r(21 + sway, 17, 9, 6)), with: .color(fur))
        ctx.fill(Path(ellipseIn: r(25 + sway, 18, 3, 4)), with: .color(dark))
        // laptop, behind the paws
        if pose.laptop { ctx.fill(Path(roundedRect: r(8, 21, 16, 8), cornerRadius: u), with: .color(Color(white: 0.3))) }
        // body
        ctx.fill(Path(ellipseIn: r(9, 15, 14, 12)), with: .color(fur))
        // ears
        let ear = pose.perk ? 0.0 : 1.0
        ctx.fill(Path(ellipseIn: r(8, 4 + ear, 5, 5)), with: .color(dark))
        ctx.fill(Path(ellipseIn: r(19, 4 + ear, 5, 5)), with: .color(dark))
        // head and mask
        ctx.fill(Path(ellipseIn: r(7, 6, 18, 14)), with: .color(fur))
        ctx.fill(Path(roundedRect: r(8.5, 11, 15, 4.5), cornerRadius: 2 * u), with: .color(dark))
        ctx.fill(Path(ellipseIn: r(12, 14, 8, 5)), with: .color(light))

        // eyes
        let lx = 12.0 + Double(pose.look) * 0.8, rx = 18.0 + Double(pose.look) * 0.8
        let eye = (pose.eyes == .wide ? 2.4 : 1.8)
        for x in [lx, rx] {
            switch pose.eyes {
            case .closed, .happy:
                var p = Path()
                p.move(to: CGPoint(x: x * u, y: (12.8 + Double(pose.dy) * 0.5) * u))
                p.addQuadCurve(to: CGPoint(x: (x + 2) * u, y: (12.8 + Double(pose.dy) * 0.5) * u),
                               control: CGPoint(x: (x + 1) * u, y: ((pose.eyes == .happy ? 11 : 13.8) + Double(pose.dy) * 0.5) * u))
                ctx.stroke(p, with: .color(light), lineWidth: 0.7 * u)
            case .open, .wide, .sad:
                ctx.fill(Path(ellipseIn: r(x, 11.8, eye, pose.eyes == .sad ? eye * 0.7 : eye)), with: .color(light))
            }
        }
        // nose and mouth
        ctx.fill(Path(ellipseIn: r(15, 16, 2, 1.4)), with: .color(dark))
        switch pose.mouth {
        case .none: break
        case .open: ctx.fill(Path(ellipseIn: r(15, 17.6, 2, 1.8)), with: .color(dark))
        case .smile, .frown:
            var p = Path()
            let y = 18.0 + Double(pose.dy) * 0.5
            p.move(to: CGPoint(x: 14.5 * u, y: y * u))
            p.addQuadCurve(to: CGPoint(x: 17.5 * u, y: y * u),
                           control: CGPoint(x: 16 * u, y: (pose.mouth == .smile ? y + 1.6 : y - 1.4) * u))
            ctx.stroke(p, with: .color(dark), lineWidth: 0.6 * u)
        }
        // paws
        if pose.armsUp {
            ctx.fill(Path(ellipseIn: r(5, 14, 4, 4)), with: .color(dark))
            ctx.fill(Path(ellipseIn: r(23, 14, 4, 4)), with: .color(dark))
        } else {
            ctx.fill(Path(ellipseIn: r(10, 25, 4, 3)), with: .color(dark))
            ctx.fill(Path(ellipseIn: r(18, 25, 4, 3)), with: .color(dark))
        }
        // props
        let glyph: String? = switch pose.prop {
        case .none: nil
        case .sound: "))"
        case .sound2: ")))"
        case .question: "?"
        case .spark: "*"
        case .sweat: "'"
        case .z: "z"
        }
        if let glyph {
            ctx.draw(Text(verbatim: glyph).font(.system(size: 6 * u, weight: .bold, design: .rounded)).foregroundColor(.accentColor),
                     at: CGPoint(x: 27 * u, y: 6 * u))
        }
    }
}
