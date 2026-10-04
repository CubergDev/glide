# CI/CD, checks and reviews: setup

Prepared offline and pushed by the user's approval. No secret, variable, label, ruleset, repo setting or app was created
or changed by an agent. Everything under "You do" below is yours to do online.

## What exists

| File | What | Gate |
|---|---|---|
| `workflows/ci.yml` | Lint and tests, workflow lint, secret scan, gated macOS tests. Ends in the aggregate job `ci-ok`. | **Required** (via `ci-ok`) |
| `workflows/app-ci.yml` | SwiftUI app tests on macOS, only when `app/**` changes. | Advisory |
| `workflows/release.yml` | Tag `vX.Y.Z` to a GitHub Release (sdist and wheel). | Not a check |
| `workflows/claude-review.yml` | Claude review of PRs. | Advisory |
| `dependabot.yml` | Weekly grouped updates: `github-actions` and `uv`, at most 3 open PRs each. | Opens PRs |
| `CODEOWNERS` | `* @CubergDev` | See "Ruleset" |
| `pull_request_template.md` | Checklist from `AGENTS.md`. | Human |
| `../.gitleaks.toml` | Secret-scan config. | Used by `secret-scan` |

Codex review has no file: it is native (below).

## Checks and which are required

Check names as GitHub shows them (job names):

| Check | Workflow | Required? | Notes |
|---|---|---|---|
| `ci-ok` | CI | **Yes, the only required check** | Succeeds only if every job below succeeded; `test-macos` may also be skipped. |
| `lint-test` | CI | via `ci-ok` | `uv sync --locked`, `uv lock --check`, `ruff check`, `ruff format --check`, offline pytest, Python 3.13. |
| `test-py312` | CI | via `ci-ok` | Offline pytest on Python 3.12, the declared floor (`requires-python >=3.12`). |
| `actionlint` | CI | via `ci-ok` | Lints `.github/workflows` with a pinned release binary, sha256-checked. |
| `secret-scan` | CI | via `ci-ok` | gitleaks, PR commits (or pushed commits on `main`). |
| `macos-gate` | CI | via `ci-ok` | Cheap Linux job that decides whether macOS runs. |
| `test-macos` | CI | via `ci-ok`, skip allowed | Runs on push to `main`, manual dispatch, a PR with label `ci-macos`, or a PR that changes `glide/computer/**`, `tests/conftest.py`, `pyproject.toml`, `uv.lock` or `ci.yml`. The diff is `git diff --name-only HEAD^1 HEAD` on the PR merge commit. |
| `app-test` | App CI | No (advisory) | Path-filtered, so a required check would hang as "pending" on PRs that do not touch `app/`. |
| `review` | claude-review | No (advisory) | Needs the secret below; a green no-op without it. |

To require another check later, either add it to `needs:` of `ci-ok` in `ci.yml` (preferred: the ruleset never changes)
or add its name in the ruleset. Never require a path-filtered or label-gated check by name.

macOS minutes cost about ten times the Linux rate on a private repo. The repo is private, so `test-macos` runs only in
the cases above; use the label `ci-macos` on a PR when you want it.

## Ruleset (you change it, not an agent)

`main` requires a PR, `ci-ok` and squash merge. Nothing here needs more. Notes:

- Do not enable "Require review from Code Owners": with `* @CubergDev` as the only owner and you as the author, GitHub
  would block your own PRs. `CODEOWNERS` then only records ownership.
- If you add a tag ruleset, it must allow you to create `v*.*.*` tags (the release flow below).
- The required check is picked by name (`ci-ok`). If it ever shows "Expected - waiting", the name or the event
  (`pull_request`) does not match.

## You do (online, once)

1. **Secret** (repo, Settings, Secrets and variables, Actions): `CLAUDE_CODE_OAUTH_TOKEN`. Create the token in your own
   terminal with `claude setup-token` (it prints the token, so not through an agent). Do not also set `ANTHROPIC_API_KEY`:
   an API key outranks the token and switches the run to API billing. No other secret is used by any workflow.
2. **Labels** (create them; GitHub does not auto-create labels): `ci-macos`, `claude-review`, `no-claude-review`,
   `no-ai-review`.
3. **Variable** (optional): `CLAUDE_REVIEW_DISABLED=true` pauses Claude review.
4. **Settings, Actions, General**: keep the fork-PR options that send secrets or write tokens off. Workflows here use no
   `pull_request_target`, and fork PRs get no secrets (Claude review skips them).
5. **Dependabot** (optional): Settings, Advisories or Code security, turn on Dependabot alerts and security updates. The
   version updates in `dependabot.yml` need nothing else.
6. Merge the PR for this branch, then check the "Still to check online" list.

## How to cut a release

The release is a GitHub Release with the sdist and wheel attached. Nothing goes to PyPI or anywhere else.

1. On a branch, change `version` in `pyproject.toml` (for example `0.2.0`) and run `uv lock`. `uv.lock` records the
   project's own version, so without this `uv sync --locked` fails in CI.
2. Open a PR, wait for `ci-ok`, squash merge to `main`.
3. Update your checkout to `main`, then tag that commit and push only the tag (you run these):
   ```
   git switch main && git pull
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
4. `release.yml` runs. `verify` checks that the tag is `vMAJOR.MINOR.PATCH`, equals `v` plus the `pyproject.toml`
   version, and points at a commit on `main`; then runs lint, `uv lock --check`, offline tests and `uv build`.
   `release` (the only job with `contents: write`) then runs `gh release create` with generated notes.
5. If `verify` fails, nothing is released. Fix on a branch, merge, delete the bad tag
   (`git push origin :refs/tags/vX.Y.Z` and `git tag -d vX.Y.Z`), tag again. Edit the Release notes on GitHub if wanted.

## Claude review (advisory)

`claude-review.yml` is the only Claude workflow. It runs on same-repo, non-draft PRs (opened, reopened, ready for review,
every push, or label `claude-review` added), reviews with read-only tools, and posts inline comments plus one summary.
It reads `AGENTS.md` and its `## Code Review Rules` section if present (the proposal below), always from the pre-PR copy.

- Pause: variable `CLAUDE_REVIEW_DISABLED=true`, or PR label `no-claude-review` / `no-ai-review`.
- Re-run: remove and re-add the label `claude-review`.
- Every push spends subscription usage. Drop `synchronize` from the trigger list if that is too much.
- No model flag: the model is the action's default. To pin one, add `--model <id>` to `claude_args` from a decision recorded
  outside the workflow.

## Codex review (native: its own login, no workflow, no secret)

Steps (from the Codex GitHub docs, read 2026-10-04; check them again if the UI differs):

1. Sign in to Codex with your account and connect GitHub, giving it access to this repository.
2. Open the Codex code-review settings, select the repository and turn on **Automatic review** to get one review per PR.
   Skip this if you prefer manual reviews.
3. Manual trigger: comment `@codex review` on a PR. Codex reacts and posts findings, focused on the most serious issues.
4. Codex reads `## Code Review Rules` sections in `AGENTS.md` (root for the whole repo). `AGENTS.md` is your file and was
   not edited; the proposal is below.

## Proposal: `## Code Review Rules` for AGENTS.md

Not applied. Paste the block into `AGENTS.md` if you want it; both Codex and the Claude review then use it.

````markdown
## Code Review Rules

Flag, with file and line and a concrete fix:

- Anything that takes over the machine without approval (screen, input, apps, AppleScript, capture, local model
  servers), or a test that can reach it. New machine-reaching calls must be added to the guard in `tests/conftest.py`
  in the same change. A weakened guard is always a finding.
- A key, header or request body in an error message, log, file or prompt. Adapters raising anything but `ProviderError`.
- A model id, endpoint, price or voice id in code or in a workflow (they are configuration).
- A fallback that is not a visible `SwitchEvent`.
- An attempted action treated as a verified effect, or a write with an unknown outcome that is replayed.
- Stored run data that includes utterances, typed text, captured content or raw URLs without detailed recording opted in.
- Page, task or provider text used as instructions, or interpolated into a shell command or a prompt.
- Workflow changes: broader `permissions`, `pull_request_target`, an action not pinned by full SHA, a secret exposed to a
  fork, a job without a timeout, a network call in tests.

Expected, do not flag: style and formatting that `ruff` enforces; placeholder keys in `tests/`; files under
`consolidation-inputs/` and `review-evidence/` (private, ignored).
````

## Pinned versions and how they move

Every third-party action is pinned to a full commit SHA with its tag in a comment. Read-only `gh api` lookups on 2026-10-04
(`git/ref/tags/<tag>`, then `git/tags/<sha>` when the tag is annotated):

| Action | Tag | Commit |
|---|---|---|
| `actions/checkout` | v7.0.1 | `3d3c42e5aac5ba805825da76410c181273ba90b1` |
| `astral-sh/setup-uv` | v10.2.0 | `c18668ad3cf93ea998bef934396af7bb5c839dc7` |
| `anthropics/claude-code-action` | v1.0.240 | `ed670b4cf9de2a5a570d130d2f6197b9e543cd64` (annotated tag, dereferenced) |
| `gitleaks/gitleaks-action` | v2.3.9 | `ff98106e4c7b2bc287b24eaf42907196329070c7` |
| `actions/upload-artifact` | v4.6.2 | `ea165f8d65b6e75b540449e92b4886f43607fa02` |
| `actions/download-artifact` | v4.3.0 | `d3f86a106a0bac45b974a628896c90dbdf5c8093` (pair with upload v4.x) |

Dependabot (`github-actions`) updates these SHAs and their comments. It does not touch the plain values in the workflows'
`env:` blocks: `UV_VERSION` (0.12.22), `ACTIONLINT_VERSION` and `ACTIONLINT_SHA256` (1.7.12), `GITLEAKS_VERSION` (8.30.1).
Bump those by hand; for actionlint, take the sha256 of the new `linux_amd64` tarball from the release's asset digest and
confirm it against the release's `checksums.txt`.

## Secret scan

`secret-scan` uses `gitleaks/gitleaks-action` with no license. Assumption: the repository belongs to the personal account
`CubergDev` (type User, confirmed), for which the action needs no `GITLEAKS_LICENSE`. If the repo ever moves to an
organization, the job fails until that secret exists.

`.gitleaks.toml` keeps every default rule and relaxes only `generic-api-key`, only under `tests/`, only for secrets that
contain an obvious placeholder marker (`0123456789`, `abcdef`, `secret`, `test`, ...). Tested locally with gitleaks 8.30.1:
0 findings over this branch's whole history, and a random key placed in `tests/` is still reported.
The config needs gitleaks 8.25 or newer, so keep `GITLEAKS_VERSION` at or above that.

## Verified here, and not

Verified offline in the worktree: `actionlint` clean on all workflows; the exact commands the jobs run (`uv sync --locked`,
`uv lock --check`, `ruff check`, `ruff format --check`, pytest on Python 3.13 and 3.12: 1544 passed, 1 xfailed; `uv build`
builds the sdist and wheel); the `ci-ok` and macOS-gate logic against sample inputs; gitleaks over history.

Not verified (needs GitHub or a macOS runner): that any workflow actually runs green on GitHub; the `secret-scan` action
download of gitleaks; `test-macos` on a runner; `app-test` (the `app/` directory is not on this branch, it lives on
`consolidation/app-swiftui`, so `app-ci.yml` does nothing until it merges, and `app/scripts/test.sh` was not run here);
`release.yml` end to end (no tag was pushed); Dependabot's uv update of `uv.lock` (the uv ecosystem is documented as
generally available; its first PR will show it); Codex steps (taken from its docs, not tried).

## Still to check online

1. The first PR: `ci-ok` and every job behind it are green, and the ruleset's required check is `ci-ok`.
2. `secret-scan` passes with the pinned gitleaks and the config.
3. Claude review: inline comments work with `github.token`. If the Claude GitHub App is installed and they do not, add
   `id-token: write` to the review job and drop the `github_token` line.
4. Label a PR `ci-macos` once to confirm `test-macos` runs and passes.
5. The first Dependabot PRs: grouped, at most 3 open, `uv.lock` updated.

## Hardening the owner must do (settings, not files)

These come from the independent security review of the workflows. They are repository settings, so only you change them.

1. **Required check source.** In the `main` ruleset, set the required check `ci-ok` with its integration pinned to
   *GitHub Actions*. Otherwise a PR can add a workflow with a job named `ci-ok` that always succeeds.
2. **Release gate.** Settings > Environments > `release`: add yourself as a required reviewer and restrict deployment
   tags to `v*.*.*`. The release job declares this environment, so it cannot run unreviewed. Consider a tag ruleset
   that restricts who can create `v*` tags. Without it, anyone with push access can tag a commit that edits the release
   checks out of the workflow file.
3. **Claude review is read-only on purpose.** The agent has the inline-comment tool, `gh pr view`/`gh pr diff` and
   read-only `git`; it has no `gh pr comment`, no `gh api`, no `/proc` and no `.git` access, so a prompt-injected PR
   cannot post the job's environment or token. If you add a tool, re-check that it cannot write free text.

## Reviewer model and effort

The Claude review runs **Opus 5.5 at high effort** on every non-fork, non-draft pull request (owner's request, 4 Oct 2026).
The model id is the default of the repository variable `CLAUDE_REVIEW_MODEL` (workflow `claude-review.yml`): to change it,
set that variable (Settings > Secrets and variables > Actions > Variables); no file edit is needed. Effort is `--effort high`
in the same file. Every push to a PR re-reviews and spends subscription usage; the large stacked PRs cost the most.
