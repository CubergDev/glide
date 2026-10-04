"""The Claude review workflow keeps its promises: nothing readable holds a token, the agent cannot read arbitrary files,
and no model id is written into the file.

The workflow is YAML and this repository has no YAML parser, so these are plain-text checks on the lines that carry
each promise. `actionlint` (run in CI) checks the syntax; these keep the security choices from being edited away.
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
REVIEW = (WORKFLOWS / "claude-review.yml").read_text(encoding="utf-8")


def _arg(flag: str) -> str:
    """The quoted value of a `claude_args` flag, for example --allowedTools."""
    match = re.search(rf'--{flag} "([^"]*)"', REVIEW)
    assert match, f"{flag} is missing from claude_args"
    return match.group(1)


def _steps_using(action: str) -> list[str]:
    """The text of every step that `uses` the action (a step runs from its `- name:` line to the next one)."""
    steps = re.split(r"(?m)^      - name:", REVIEW)
    return [step for step in steps if f"uses: {action}@" in step]


def test_the_checkout_leaves_no_credential_behind():
    (checkout,) = _steps_using("actions/checkout")
    assert re.search(r"(?m)^\s+persist-credentials: false\s*$", checkout)


def test_the_agent_cannot_read_arbitrary_files_through_git():
    allowed = _arg("allowedTools")
    assert "git blame" not in allowed  # `git blame --contents <file>` prints any file
    assert not re.search(r"Bash\((?!gh pr (?:diff|view):|git (?:show|log):)", allowed.replace("Bash(", "\nBash(")), allowed


def test_the_runner_temp_directory_is_denied_to_the_agent():
    denied = _arg("disallowedTools")
    assert "Read(/${{ runner.temp }}/**)" in denied  # the `//` form is an absolute path
    assert "Read(.git/**)" in denied and "Read(//proc/**)" in denied


def test_the_prefetched_threads_file_is_outside_the_denied_directories():
    fetch = re.search(r"inline-comments\.jsonl", REVIEW)
    assert fetch, "the workflow must fetch the inline review comments for the agent"
    for path in re.findall(r"[\w${}/.~-]*inline-comments\.jsonl", REVIEW):
        assert "RUNNER_TEMP" not in path and "runner.temp" not in path and ".claude" not in path, path
    prompt_path = re.search(r"(/home/runner/[\w/.-]*inline-comments\.jsonl)", REVIEW)
    assert prompt_path, "the prompt must name the file the agent reads"
    assert prompt_path.group(1).startswith("/home/runner/review-context/")


def test_the_agent_has_no_way_to_call_the_api():
    assert "Bash(gh api*)" in _arg("disallowedTools")
    assert "gh api" not in _arg("allowedTools")


def test_every_pr_view_asks_for_explicit_fields():
    """A bare `gh pr view` asks for the check rollup, which this job's token cannot read."""
    for line in REVIEW.splitlines():
        if "`gh pr view" in line:
            assert "--json" in line or "gh pr view:*" in line, line.strip()


def test_the_job_does_not_ask_for_checks_it_never_reads():
    job = REVIEW[REVIEW.index("    permissions:") :]
    assert "checks:" not in job.split("steps:")[0]


def test_no_model_id_is_written_into_the_workflow():
    args = REVIEW[REVIEW.index("claude_args:") :]
    assert "--model" in args and "vars.CLAUDE_REVIEW_MODEL" in args
    assert not re.search(r"claude-[a-z0-9]", args, re.IGNORECASE), "a model id belongs in the repository variable"
    assert not re.search(r"(?i)\|\|\s*'[^']+'", args.split("--effort")[0]), "no fallback literal for the model"
