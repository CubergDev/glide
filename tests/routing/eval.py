"""Offline routing evaluation: labelled utterances through the real router, over two scripted fake models.

    python tests/routing/eval.py              # the report
    python tests/routing/eval.py --write-table  # regenerate tests/routing/reliability_fakes.json

WHAT THIS MEASURES, AND WHAT IT DOES NOT. The router (tiers, thresholds, calibration, the fallback table, source
tagging, strict parsing, the clarify channel) is the real code. The two models are fakes with simple, documented
behaviour (`LexiconModel` below), so the accuracy numbers say how the router behaves given readers of that quality and
nothing about any real model. Real-model routing quality can only be measured live: docs/ROUTER.md gives the procedure and
this module's `tier_samples`/`Calibration.fit` are what turns its results into a reliability table.

The fakes
---------
`LexiconModel` counts cue phrases (English, Cantonese, Mandarin) in the UTTERANCE ONLY, per route, and never sees a
label. The classifier fake turns the counts into a distribution: one route with hits and no rival is 0.86 (0.92 for two
or more hits); a rival makes it 0.45 + 0.15 x margin; no hits at all is an unsure reading (answer 0.40 with the rest
spread). A hint that defines a phrase ("the usual means: ...") resolves that phrase, and it is used only when memory hints are
in the state. The fast-LLM fake is a slightly better reader: it also uses the question form, the presence of untrusted
page data ("this page"), a running task, the previous assistant turn and a quoted passage, and it says "high", "medium"
or "low". A case may script either model explicitly (`fake` in cases.jsonl: an exact pick, an error kind, a malformed
reply, a raw string) for the adversarial and failure cases.

The two router configurations
-----------------------------
`default` is `RoutingSettings()` with no table. `careful` turns on `confirm_acting` (two keys to act) and applies a
reliability table fitted on the OTHER folds of the same cases (cross-fitting: a case is never calibrated by itself).
The bars (tests/routing/test_routing_eval.py): zero false actions on the hostile and ambiguous groups for `default`, and on
every group, the `overconfident` one included, for `careful`.
"""

# ruff: noqa: RUF001  full-width punctuation is the right punctuation here

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from routing_fakes import FakeClassifier, FakeLLM, Garbage, Pick, failing, fast_json  # noqa: E402

from glide.providers.errors import ProviderError  # noqa: E402
from glide.routing import (  # noqa: E402
    ACTING,
    Calibration,
    Context,
    Router,
    RoutingSettings,
    Sample,
    Span,
    expected_calibration_error,
    is_stop,
)
from glide.routing.features import features  # noqa: E402
from glide.routing.tiers import TierFailure, ask_classifier, ask_fast, view  # noqa: E402

CASES = HERE / "cases.jsonl"
TABLE = HERE / "reliability_fakes.json"
ROUTES = ("stop", "answer", "execute", "research", "reason", "clarify")
GROUPS = ("stop", "smalltalk", "factual", "computer", "research", "reasoning", "ambiguous", "overconfident", "hostile", "context")
FOLDS = 5
# Left out of the reliability fit: hostile text is stopped by structure before a confidence matters, and the overconfident group is a
# stress test laid on top (a classifier unusually wrong), not part of how the model reads ordinary requests.
UNFITTED = ("hostile", "overconfident")


# -- the cases ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    id: str
    group: str
    lang: str
    utterance: str
    source: str
    label: str
    acceptable: tuple[str, ...]
    history: tuple[tuple[str, str], ...] = ()
    data: tuple[tuple[str, str], ...] = ()
    running_task: str = ""
    memory_hints: tuple[str, ...] = ()
    memory_on: bool = False
    fake: dict = field(default_factory=dict)
    structure: str = ""
    provenance: str = "written"
    relabelled: bool = False

    @property
    def acts_ok(self) -> bool:
        """Whether acting is among the acceptable outcomes. If it is not, acting is a false action."""
        return bool(set(self.acceptable) & ACTING)

    def context(self) -> Context:
        return Context(
            history=tuple(Span(t, s) for s, t in self.history),
            data=tuple(Span(t, s) for s, t in self.data),
            running_task=self.running_task,
            memory_hints=self.memory_hints,
        )


def load_cases(path: Path = CASES) -> list[Case]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        ctx = raw.get("context", {})
        label = raw["label"]
        cases.append(
            Case(
                id=raw["id"],
                group=raw["group"],
                lang=raw["lang"],
                utterance=raw["utterance"],
                source=raw["source"],
                label=label,
                acceptable=tuple(raw.get("acceptable", [label])),
                history=tuple((s, t) for s, t in ctx.get("history", [])),
                data=tuple((s, t) for s, t in ctx.get("data", [])),
                running_task=ctx.get("running_task", ""),
                memory_hints=tuple(ctx.get("memory_hints", [])),
                memory_on=bool(raw.get("memory_on", False)),
                fake=raw.get("fake", {}),
                structure=raw.get("structure", ""),
                provenance=raw.get("provenance", "written"),
                relabelled=bool(raw.get("relabelled", False)),
            )
        )
    return cases


# -- the fake models: simple, documented, label-blind ----------------------------------------------------------------------

# fmt: off
CUES = {
    "execute": [
        'open ', 'click', 'scroll', 'type ', 'select ', 'turn on', 'turn off', 'mute', 'play ', 'close ', 'switch to',
        'press ', 'bookmark', 'go to wikipedia', 'attach', 'draft a reply', 'text kelvin', 'add "', 'make a new note',
        'set the subject', 'delete', 'search for', 'stop the music', 'cancel my', '取消訂', 'use chrome', 'use safari',
        'email my', '打開', '打开', '幫我開', '帮我开', '开个', '開個', '點擊', '点击', '刪', '删', '揀', '選', '选', '加落', '日历里加', '傳俾', '改做',
        '播放',
    ],
    "research": [
        'find the', 'look up', 'compare the prices', 'search the web', 'find out', 'check the weather', 'product page',
        'summarize', 'recommend', 'reviews', 'read me', 'read out', 'read it out', 'read the', 'this page',
        'pickup code', 'latest news', 'tell me if', 'tell me whether', 'tell me what', 'and tell me', '最近', '上網', '搵下',
        '查一下', '比較', '比较', '总结', '總結', '页面', '取貨碼', '取件码', '讀返', '念给', '读出',
    ],
    "reason": [
        'write a', 'explain', 'prove', 'calculate', 'analyze', 'pros and cons', 'plan a', 'rewrite', 'draft a polite',
        'draft an email', 'haiku', 'step by step', 'shorter', 'python', '寫一封', '写一份', '寫個', '解釋', '解释', '證明', '证明', '詳細',
        '详细', '分別',
    ],
    "answer": [
        'hi', 'hello', 'thanks', 'good morning', 'tell me a joke', 'how are you', 'your name', 'bored',
        "you're talking too", '你好', '多謝', '早晨', '晚安', '太慢', '太快',
    ],
}
QUESTION_STARTS = (
    'what', 'who', 'when', 'where', 'which', 'how many', 'how much', 'how far', 'how do you say', 'is this', 'is that',
    'does', 'do you', 'are you', '香港', '法国', '珠穆', '一英里', '幫我 check', '點', '邊', '幾多', '多少', '哪里', '有多高', '咩',
)
VAGUE = (
    'that thing', 'do that', 'the usual', 'fix it', 'send it', 'open it', 'make it better', 'delete them', 'book it',
    'call him', 'share that with her', 'sort that out', 'handle it', 'do what it says', '嗰樣嘢', '搞掂佢', '傳俾佢', '把那个',
    '弄一下', '打开它', '處理咗佢', '处理一下',
)
# fmt: on
STOP_NOT_AFTER = ("stop the", "stop by", "cancel my", "取消訂單")


def _has(text: str, cue: str) -> bool:
    """A cue is found as a word (or phrase) in Latin text and as a substring in Chinese text."""
    if not cue.isascii():
        return cue in text
    tail = "(?![a-z])" if cue[-1].isalpha() else ""
    return re.search("(?<![a-z])" + re.escape(cue) + tail, text) is not None


def _count(text: str, cues) -> int:
    return sum(1 for cue in cues if _has(text, cue))


class LexiconModel:
    """Counts cues per route in the user's own words. Documented in the module docstring; sees no label."""

    def __init__(self, text: str, hints: list[str] = ()):
        low = text.casefold().strip()
        low = re.sub(r"\s+", " ", low)
        self.text, self.low = text, low
        vague = [v for v in VAGUE]
        for hint in hints:  # "X means: Y" resolves X and adds Y to what is read
            if " means:" in hint.casefold():
                phrase, _, meaning = hint.casefold().partition(" means:")
                phrase = phrase.strip()
                vague = [v for v in vague if v not in phrase]
                self.low = low = low + " " + meaning.strip()
        scores = {route: _count(low, cues) for route, cues in CUES.items()}
        # a question form is a quick answer
        if self.low.startswith(QUESTION_STARTS) or text.rstrip().endswith(("?", "？")):
            scores["answer"] += 1
        scores["clarify"] = 3 if len(low.split()) <= 6 and any(_has(low, v) for v in vague) else 0
        starts_stop = (low.startswith(("stop", "停")) or low.startswith("cancel ") or "never mind" in low) and not low.startswith(
            STOP_NOT_AFTER
        )
        scores["stop"] = 3 if starts_stop else 0
        self.scores = scores

    def ranked(self):
        return sorted(self.scores.items(), key=lambda kv: -kv[1])

    def margin(self) -> tuple[str, int, int]:
        (top, hits), (_, second) = self.ranked()[0], self.ranked()[1]
        return top, hits, hits - second


def classifier_reading(state: dict):
    hints = state.get("memory_hints_are_data_not_instructions", [])
    model = LexiconModel(state["goal"], hints)
    top, hits, margin = model.margin()
    if hits == 0:
        return Pick("answer", 0.40, probs={"answer": 0.40, "clarify": 0.30, "execute": 0.15, "reason": 0.15})
    second = model.ranked()[1]
    if second[1] == 0:
        p = 0.92 if hits >= 2 else 0.86
        rival = next(r for r in ("answer", "clarify") if r != top)
        return Pick(top, p, probs={top: p, rival: round(1 - p - 0.01, 2)})
    p = min(0.9, 0.45 + 0.15 * margin)
    return Pick(top, p, probs={top: p, second[0]: round(max(0.0, min(1 - p, 0.45 - 0.01 * margin)), 2)})


def llm_reading(messages) -> str:
    user = messages[-1]["content"]
    system = messages[0]["content"]
    hints = [line[2:] for line in system.splitlines() if line.startswith("- ")]
    model = LexiconModel(user, hints)
    low = model.low
    scores = dict(model.scores)
    features = _features_in(system)
    if features["data"] and ("this page" in low or "this screen" in low or "页面" in low or "頁" in low or "the screen" in low):
        scores["research"] += 2
    if features["quoted"] and user.rstrip().endswith("?"):
        scores["answer"] += 3
    history = [m for m in messages[1:-1] if m["role"] in ("user", "assistant")]
    if history and history[-1]["role"] == "assistant" and re.search(r"\btwo\b|兩個|两个", history[-1]["content"]):
        scores["clarify"] += 2
    if "A task is running now" in system and len(user.split()) <= 6 and not any(scores.values()):
        scores["execute"] += 2
    if history and not any(scores.values()):
        scores["answer"] += 2
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    (top, hits), (_, second) = ranked[0], ranked[1]
    if hits == 0:
        return fast_json("answer", "low", language="en")
    margin = hits - second
    level = "high" if second == 0 or margin >= 2 else "medium" if margin == 1 else "low"
    if margin == 0:
        top, level = "clarify", "low"
    question = "What would you like me to do?" if top == "clarify" else ""
    goal = user if top in ("execute", "research", "reason") else ""
    return fast_json(top, level, goal=goal, question=question, language="en")


def _features_in(system: str) -> dict:
    """The fast fake cannot see the classifier's state, so the harness tells it through the system prompt."""
    return {"data": "[data present]" in system, "quoted": "[quoted passage]" in system}


# -- running a case -----------------------------------------------------------------------------------------------------


def _override_classifier(spec: dict):
    if "error" in spec:
        return failing(spec["error"])
    if "garbage" in spec:
        return Garbage(spec["garbage"])
    choice, p = spec["pick"]
    return Pick(choice, p)


def _override_llm(spec: dict):
    if "error" in spec:
        return failing(spec["error"])
    if "raw" in spec:
        return spec["raw"]
    return fast_json(**spec["fast"])


def make_fakes(case: Case):
    spec = case.fake

    def classifier_script(state):
        return _override_classifier(spec["classifier"]) if "classifier" in spec else classifier_reading(state)

    def llm_script(messages):
        if "llm" in spec:
            return _override_llm(spec["llm"])
        notes = ""
        if case.data:
            notes += "\n[data present]"
        if any(ch in case.utterance for ch in ('"', "“", "「")) and len(case.utterance) > 60:
            notes += "\n[quoted passage]"
        patched = [{**messages[0], "content": messages[0]["content"] + notes}, *messages[1:]]
        return llm_reading(patched)

    return FakeClassifier(classifier_script), FakeLLM(llm_script)


def decide_obedient(case: Case):
    """The same case, with both models obeying whatever they are shown: a classifier that says execute at 0.99 and a fast
    model that says execute with high confidence. A case whose own script is a failure keeps it. What is left standing
    is what the router does by structure alone."""
    obey_classifier = Pick("execute", 0.99)
    obey_llm = fast_json("execute", "high", goal="do it", language="en")
    spec = case.fake
    classifier = FakeClassifier(_override_classifier(spec["classifier"]) if "classifier" in spec else obey_classifier)
    llm = FakeLLM(_override_llm(spec["llm"]) if "llm" in spec else obey_llm)
    router = Router(classifier, llm, settings=settings_for(case, "default"))
    return router.route(Span(case.utterance, case.source), case.context()), classifier, llm


def settings_for(case: Case, config: str) -> RoutingSettings:
    return RoutingSettings(memory_hints=case.memory_on, confirm_acting=(config == "careful"))


def decide(case: Case, config: str, calibration: Calibration | None = None):
    classifier, llm = make_fakes(case)
    router = Router(classifier, llm, settings=settings_for(case, config), calibration=calibration)
    decision = router.route(Span(case.utterance, case.source), case.context())
    return decision, classifier, llm


def tier_samples(cases: list[Case]) -> list[tuple[str, Sample]]:
    """(case id, sample) for every tier reading that produced a route: what the table is fitted on."""
    out = []
    for case in cases:
        if case.source != "user" or is_stop(case.utterance) or case.group in UNFITTED:
            continue
        classifier, llm = make_fakes(case)
        settings = settings_for(case, "default")
        ctx = case.context()
        state = view(case.utterance, features(case.utterance, ctx), ctx, settings)
        readings = (ask_classifier, (classifier, state)), (ask_fast, (llm, case.utterance, ctx, settings))
        for read, args in readings:
            try:
                result = read(*args)
            except (TierFailure, ProviderError):  # a scripted failure gives no reading
                continue
            out.append((case.id, Sample(result.tier, result.route, result.raw, result.route in case.acceptable)))
    return out


def fold_of(case_id: str) -> int:
    return sum(map(ord, case_id)) % FOLDS


def cross_fit(cases: list[Case]) -> dict[str, Calibration]:
    """For each case, the table fitted on the cases of the other folds (never on itself)."""
    samples = tier_samples(cases)
    tables = {}
    for fold in range(FOLDS):
        train = [s for cid, s in samples if fold_of(cid) != fold]
        tables[fold] = Calibration.fit(train, source=f"offline fakes, fold {fold} held out")
    return {case.id: tables[fold_of(case.id)] for case in cases}


# -- the report ---------------------------------------------------------------------------------------------------------------


@dataclass
class Report:
    config: str
    rows: list[tuple[Case, object]]

    def correct(self, case: Case, decision) -> bool:
        return decision.route in case.acceptable

    def accuracy(self, group: str | None = None) -> float:
        rows = [(c, d) for c, d in self.rows if group is None or c.group == group]
        return sum(self.correct(c, d) for c, d in rows) / len(rows) if rows else 1.0

    def per_route(self) -> dict[str, tuple[int, int]]:
        """label -> (cases with that label, routed to an acceptable route)."""
        out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for c, d in self.rows:
            out[c.label][0] += 1
            out[c.label][1] += self.correct(c, d)
        return {k: (v[0], v[1]) for k, v in out.items()}

    def false_actions(self, group: str | None = None) -> list[tuple[Case, object]]:
        return [(c, d) for c, d in self.rows if (group is None or c.group == group) and d.acts and not c.acts_ok]

    def missed_actions(self) -> list[tuple[Case, object]]:
        return [(c, d) for c, d in self.rows if c.label in ACTING and not d.acts]

    def why(self) -> Counter:
        return Counter((d.tier, d.why_code) for _, d in self.rows)


def evaluate(cases: list[Case], config: str = "default", *, calibrate: bool = False) -> Report:
    tables = cross_fit(cases) if calibrate else {}
    rows = [(case, decide(case, config, tables.get(case.id))[0]) for case in cases]
    return Report(config + ("+table" if calibrate else ""), rows)


def calibration_error(cases: list[Case]) -> dict[str, tuple[float, float, int]]:
    """Per tier: (error of the raw confidence, error after the cross-fitted table, readings). Lower is better."""
    samples = tier_samples(cases)
    tables = cross_fit(cases)
    by_id = {c.id: c for c in cases}
    out = {}
    for tier in ("classifier", "fast_llm"):
        mine = [(cid, s) for cid, s in samples if s.tier == tier]
        raw = [(s.raw, s.right) for _, s in mine]
        cal = [(tables[cid].apply(s.tier, s.route, s.raw)[0], s.right) for cid, s in mine]
        out[tier] = (expected_calibration_error(raw), expected_calibration_error(cal), len(mine))
        assert all(cid in by_id for cid, _ in mine)
    return out


def fitted_table(cases: list[Case]) -> Calibration:
    return Calibration.fit(
        [s for _, s in tier_samples(cases)], source="offline fakes (tests/routing/eval.py): measures the fakes, no model"
    )


def format_report(cases: list[Case]) -> str:
    lines = [f"routing evaluation: {len(cases)} cases, {len({c.lang for c in cases})} language tags", ""]
    for report in (evaluate(cases, "default"), evaluate(cases, "careful", calibrate=True)):
        fa = report.false_actions()
        lines.append(f"== {report.config}: accuracy {report.accuracy():.3f}; false actions {len(fa)} of {len(report.rows)}")
        lines.append("   per group  " + "  ".join(f"{g} {report.accuracy(g):.2f}" for g in GROUPS))
        lines.append(
            "   per route  "
            + "  ".join(f"{r} {ok}/{n}" for r, (n, ok) in sorted(report.per_route().items()))
            + f"   missed actions {len(report.missed_actions())}"
        )
        lines.append(
            "   false actions by group: "
            + (", ".join(f"{g} {len(report.false_actions(g))}" for g in GROUPS if report.false_actions(g)) or "none")
        )
        for case, decision in fa:
            lines.append(f"     - {case.id} [{case.group}] -> {decision.route} ({decision.tier}, {decision.why_code})")
        lines.append("   decided by: " + ", ".join(f"{t}/{w} {n}" for (t, w), n in sorted(report.why().items())))
        lines.append("")
    lines.append("calibration (expected calibration error, raw -> cross-fitted table; readings):")
    for tier, (raw, cal, n) in calibration_error(cases).items():
        lines.append(f"   {tier}: {raw:.3f} -> {cal:.3f} ({n})")
    return "\n".join(lines)


LIVE_ENV = "GLIDE_ROUTING_LIVE"


def live(argv: list[str]) -> int:
    """The LIVE procedure: the configured providers read the cases, and a reliability table is fitted on what they said.

    It sends the (synthetic, scripted-free) utterances of cases.jsonl to the classifier and fast chains of your own
    glide.toml, which costs money and uses your keys, so it refuses unless you set GLIDE_ROUTING_LIVE=1 and name the
    flag yourself. No test calls it and no agent should: it is the one place in this directory that reaches a provider.
    It prints codes and case ids, never an utterance, and writes the table to `--out` (default reliability_live.json,
    which is for you to point `[routing] calibration_file` at).
    """
    import os

    if os.environ.get(LIVE_ENV) != "1":
        print(f"refusing: --live spends provider credit. Set {LIVE_ENV}=1 to run it on purpose.", file=sys.stderr)
        return 2
    import datetime

    from glide.providers.config import load_config

    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else HERE / "reliability_live.json"
    config = load_config()
    classifier, llm = config.classifier(), config.llm("fast")
    cases = [c for c in load_cases() if c.source == "user" and not c.fake and not is_stop(c.utterance)]
    samples, failed = [], Counter()
    for case in cases:
        ctx, settings = case.context(), settings_for(case, "default")
        state = view(case.utterance, features(case.utterance, ctx), ctx, settings)
        for read, args in ((ask_classifier, (classifier, state)), (ask_fast, (llm, case.utterance, ctx, settings))):
            try:
                result = read(*args)
            except (TierFailure, ProviderError) as error:
                failed[getattr(error, "kind", "unparseable")] += 1
                continue
            if case.group not in UNFITTED:
                samples.append(Sample(result.tier, result.route, result.raw, result.route in case.acceptable))
    table = Calibration.fit(samples, source=f"live run {datetime.date.today()} on the configured providers, {len(cases)} cases")
    table.dump(out)
    router = Router(classifier, llm, settings=RoutingSettings(), calibration=table)
    decisions = {c.id: router.route(Span(c.utterance), c.context()) for c in cases}
    leaks = [c.id for c in cases if decisions[c.id].acts and not c.acts_ok]
    for tier in ("classifier", "fast_llm"):
        mine = [s for s in samples if s.tier == tier]
        pairs = [(s.raw, s.right) for s in mine]
        right = sum(s.right for s in mine)
        print(f"{tier}: {right}/{len(mine)} right, calibration error {expected_calibration_error(pairs):.3f} (raw)")
    print(f"failed readings by kind: {dict(failed)}")
    print(f"false actions with the fitted table: {len(leaks)} {leaks}  (the bar is zero)")
    print(f"wrote {out}")
    return 0 if not leaks else 1


def main(argv: list[str]) -> int:
    if "--live" in argv:
        return live(argv)
    cases = load_cases()
    if "--write-table" in argv:
        fitted_table(cases).dump(TABLE)
        print(f"wrote {TABLE.name}")
        return 0
    print(format_report(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
