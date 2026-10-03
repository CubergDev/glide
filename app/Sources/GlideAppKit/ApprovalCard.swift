import GlideProtocol
import SwiftUI

/// One approval. Shows the exact command as plain text and offers Approve once or Deny. Never auto-approves.
struct ApprovalCard: View {
    var model: AppModel
    var approval: PendingApproval

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(approval.request.kind.label, systemImage: "hand.raised").font(.caption.weight(.semibold))
            // The command is untrusted text from the core's task: verbatim, selectable, never interpreted.
            Text(verbatim: approval.request.command)
                .font(.system(.callout, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(6)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
            if let deadline = approval.deadline {
                TimelineView(.periodic(from: .now, by: 1)) { t in
                    let left = max(0, Int(deadline.timeIntervalSince(t.date).rounded(.up)))
                    Text(verbatim: "No answer in \(left) s counts as Deny").font(.caption2).foregroundStyle(.secondary)
                }
            }
            HStack {
                Button("Deny", role: .destructive) { model.respond(to: approval.id, .deny) }
                Spacer()
                Button("Approve once") { model.respond(to: approval.id, .approve) }
                    .buttonStyle(.borderedProminent)
            }
            .controlSize(.small)
        }
        .padding(8)
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.orange, lineWidth: 1))
    }
}
