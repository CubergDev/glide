"""Regressions for review findings against research and grounding (ids in each test's docstring)."""

import pytest
from research_fakes import QUOTE_A, SEARCH, A, B, Pages, Planner, claim, decision, literal_urls

from glide.computer.execution import grounding, planning
from glide.computer.execution.contracts import InvalidAction, Observation
from glide.computer.execution.research import Supervisor, Tools

EVIL = "https://evil.example/"


def supervisor(**kw):
    tools = Tools(Planner(), lambda *a: None, literal_urls, lambda action, observed: False)
    return Supervisor("Review the place.", "research", 24, tools=tools, search_url=SEARCH, **kw)


def read(sup, url, text, links=()):
    sup.sources.append(
        {
            "id": f"s{len(sup.sources)}",
            "origin": "page",
            "title": "T",
            "url": url,
            "text": text,
            "links": [{"url": u, "title": "x"} for u in links],
        }
    )


def test_a_url_in_a_question_is_not_authorised_by_the_users_no():
    """PR12-4175598269: only the user's own words ground an address, never the question that was put to them."""
    sup = supervisor()
    sup.tools = Tools(Planner(), lambda *a: None, literal_urls, lambda action, observed: False)
    ctx = type("Ctx", (), {"ask": lambda self, q: "no"})()
    assert sup.ask(ctx, f"Shall I open {EVIL} ?") == "no"
    observed = Observation("browser", "x", SEARCH, "t", {"t": SEARCH}, capabilities={"inspect"})
    assert EVIL.rstrip("/") not in {u.rstrip("/") for u in sup.allowed_urls(observed)}
    assert sup.replies and EVIL in sup.replies[0]  # the model still sees the exchange
    sup.ask(type("Ctx", (), {"ask": lambda self, q: f"yes, use {A}"})(), "Which?")
    assert A in sup.allowed_urls(observed)


def test_grounding_reads_the_users_words_not_the_clarifying_question():
    """PR12-4175598269: the planner's and router's literals come from the request and the replies only."""
    goal = grounding.Request(f"Open the page\n\nClarifications:\nOpen {EVIL}?\nUser: no", stated="Open the page\nno")
    assert grounding.extract(goal).urls == []
    observed = Observation("browser", "x", SEARCH, "t", {"t": SEARCH}, capabilities={"inspect"})
    assert planning._origin(EVIL) not in planning.grounded_origins(goal, "", observed, [], SEARCH)


def test_a_citation_title_cannot_carry_an_address_the_run_did_not_read():
    """PR12-4175420946: a page-chosen title is written into the answer, so it is checked like a claim."""
    sup = supervisor()
    read(sup, A, QUOTE_A)
    sup.sources[-1]["title"] = f"Visit {EVIL}"
    answer = sup.draft(decision("answer", claims=[claim("Waits are long.", "s1")]))
    assert EVIL not in answer.text


def test_an_address_is_only_accepted_whole():
    """PR12-4175612812: a prefix of a read address (a shorter host, a cut-off path) is not that address."""
    sup = supervisor()
    read(sup, "https://bank.example.com.evil.test/login", "x")
    for bad in ("https://bank.example.com", "https://b", "https://bank.example.com.evil.test/log"):
        with pytest.raises(InvalidAction):
            sup.check_addresses([f"See {bad} now."])
    sup.check_addresses(["See https://bank.example.com.evil.test/login. Also (https://BANK.example.com.evil.test/login)."])


def test_spoken_answer_names_no_address_the_written_one_keeps():
    """PR12-4175623621: the spoken version has no addresses."""
    sup = supervisor()
    read(sup, A, QUOTE_A)
    answer = sup.draft(decision("answer", claims=[claim(f"Reviews at {A} say waits are long.", "s1")]))
    assert A in answer.text and "http" not in answer.spoken_text


def test_a_reread_page_with_new_links_adds_them_to_the_allowed_addresses():
    """PR12-4175598275: identical text with different anchors still adds the destinations."""
    pages = Pages(url=SEARCH).page(SEARCH, QUOTE_A, [(A, "one")])
    sup = supervisor()
    sup.collect(pages)
    pages.pages[SEARCH]["links"].append({"url": B, "title": "two"})
    sup.collect(pages)
    assert B in sup.allowed_urls(pages.inspect()) and len(sup.sources) == 2
