# Review setup

Prepared offline, then pushed once to `main` by the user's approval. No secret, ruleset, label, variable or app was
created by an agent.

## Files

| File | What |
|---|---|
| `workflows/ci.yml` | `uv sync --locked`, ruff check and format check, offline pytest on Linux (Python 3.13); aggregate job `ci-ok`. macOS pytest is manual-dispatch only (private-repo macOS minutes cost about ten times Linux; run it from the Actions tab when you want it). No secrets. |
| `workflows/claude-review.yml` | Advisory Claude review on same-repo, non-draft PRs. Uses the one secret below. A green no-op until the secret exists. |

## Secrets (exactly one)

- `CLAUDE_CODE_OAUTH_TOKEN`: a Claude subscription token. Create it in your own terminal with `claude setup-token`
  (it prints the token, so do not do this through an agent) and store it as a repository secret. Do not also set an
  `ANTHROPIC_API_KEY` secret: an API key outranks the token and switches the run to API billing.

## Codex (native, no workflow, no secret)

Codex review runs through your own Codex login and its GitHub integration, configured on the Codex side.

- Enable it for the repository in the Codex settings (code review). Turn on automatic reviews if you want one per PR.
- Manual trigger: comment `@codex review` on the PR (taken from the archive notes; confirm against current Codex docs).
- Codex reads `AGENTS.md`. The archive relied on a `## Code Review Rules` section there; the current `AGENTS.md` has none.
  It is single-owner and was not edited here.

## Controls

- Pause Claude review: repository variable `CLAUDE_REVIEW_DISABLED=true`, or PR label `no-claude-review` / `no-ai-review`.
- Re-run: remove and re-add the label `claude-review`.
- Every push to a PR spends subscription usage (`synchronize` re-reviews). Remove it from the trigger list if that is too much.
- Settings, Actions, General: keep the fork-PR options that send secrets or write tokens off.

## Verified read-only on 2026-10-04

`actions/checkout` v7.0.1, `astral-sh/setup-uv` v10.2.0 and `anthropics/claude-code-action` v1.0.240 each resolve to the
pinned SHA, and uv 0.12.22 exists.

## Still to check online

1. Inline comments work with `github.token` on the first PR. If the Claude GitHub App is installed and they do not,
   add `id-token: write` to the review job and drop the `github_token` line.
2. The Linux CI job passes (the suite was only run on macOS here).
3. If you add branch protection or a ruleset, its required check names match `ci-ok` (and `lint-test`).
