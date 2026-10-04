"""Where text from the screen or a model may take an action: the address use_browser opens, and the fields it types into.

A destination comes from the catalog (configuration), from an address in the user's own goal, or from the writer's
proposal. A proposal is model output, built from the goal and from a history that quotes screen text, so it is checked
in code before the browser is told to open it (AGENTS.md: page, task and provider text is untrusted data). The macOS
adapter puts the address inside an AppleScript string literal, so a quote or backslash in it would end that literal.
"""

from types import SimpleNamespace

import pytest
from PIL import Image

from glide.computer import actions
from glide.computer.models import Field, Screen
from glide.computer.platform_adapter import desktop

SCREEN = Screen(image=Image.new("RGB", (2000, 1200)), scale=2.0, app="Google Chrome", field=None, url=None)


def context(goal="find the next upcoming bruno mars concert") -> actions.Context:
    return actions.Context(goal=goal, browser="Google Chrome", email="me@example.org", typesafe=None, writer=object(), history=[])


@pytest.fixture
def opened(monkeypatch):
    log = []
    monkeypatch.setattr(desktop, "activate", lambda app: log.append(("activate", app)) or True)
    monkeypatch.setattr(desktop, "open_url", lambda app, url: log.append(("open", url)) or True)
    return log


def use_browser(opened, proposal, goal="find the next upcoming bruno mars concert"):
    """What use_browser does when the classifier says `other` and the writer proposes `proposal`."""
    decision = SimpleNamespace(chosen="use_browser", site=SimpleNamespace(choice="other"))
    return actions.perform(decision, SCREEN, [], context(goal))


@pytest.mark.parametrize(
    "proposal",
    [
        'https://a.example/"&(do shell script "id")&"',  # ends the AppleScript string literal
        "https://a.example/\\x",
        "https://a.example/\x07",
        "https://user:secret@a.example/",
        "https://a.example:8443/",
        "https://a.example/?q=what+the+screen+said",  # a query carries data out
        "https://a.example/#what-the-screen-said",
        "https://192.168.1.1/admin",
        "https://printer.local/",
        "https://localhost.localdomain/",
        "https://ex\u0430mple.com/",  # with a Cyrillic a
    ],
)
def test_a_proposed_address_that_could_carry_data_or_break_out_is_refused(opened, monkeypatch, proposal):
    monkeypatch.setattr(actions, "compose_url", lambda *a: proposal)
    refusal = use_browser(opened, proposal)
    assert refusal.startswith("use_browser refused: ") and "address" in refusal
    assert opened == []  # nothing was opened, and the browser was not even brought forward


@pytest.mark.parametrize(
    "proposal",
    ["https://www.songkick.com/", "https://en.wikipedia.org/wiki/Bruno_Mars", "https://docs.python.org/3/library/"],
)
def test_a_plain_public_page_the_writer_proposes_is_opened(opened, monkeypatch, proposal):
    monkeypatch.setattr(actions, "compose_url", lambda *a: proposal)
    assert use_browser(opened, proposal) == f"opened {proposal}"
    assert opened == [("open", proposal)]


def test_an_address_the_user_typed_into_the_goal_may_carry_a_query_and_a_port(opened):
    goal = "open example.com:8443/search?q=cats"  # the user's own words, which compose_url returns without the writer
    assert use_browser(opened, "", goal) == "opened https://example.com:8443/search?q=cats"


def test_an_address_in_the_goal_that_would_end_an_applescript_string_is_still_refused(opened):
    goal = 'open example.com/"x'
    assert use_browser(opened, "", goal).startswith("use_browser refused: ")
    assert opened == []


def test_a_catalog_site_needs_no_check(opened):
    decision = SimpleNamespace(chosen="use_browser", site=SimpleNamespace(choice="github"))
    assert actions.perform(decision, SCREEN, [], context()) == "opened https://github.com/"


def field(label="Search", placeholder="", role="AXTextField") -> Field:
    return Field(role=role, label=label, placeholder=placeholder, value="", x=10, y=20, w=200, h=30, ref=None)


@pytest.fixture
def typed(monkeypatch):
    log = []
    monkeypatch.setattr(desktop, "clear_field", lambda: log.append("clear"))
    monkeypatch.setattr(desktop, "type_text", lambda text: log.append(text))
    monkeypatch.setattr(desktop, "press", lambda key, command=False: log.append(f"press {key}"))
    monkeypatch.setattr(actions, "compose_text", lambda *a: SimpleNamespace(text="quarterly report", submit=True))
    return log


@pytest.mark.parametrize("chosen", ["type_text", "type_email"])
@pytest.mark.parametrize(
    "label, placeholder",
    [
        ("Password", ""),
        ("", "Enter your PIN"),
        ("Card number", ""),
        ("One-time code", ""),
        ("API key", ""),
        ("Recovery codes", ""),
    ],
)
def test_nothing_is_typed_into_a_field_that_asks_for_a_credential(typed, chosen, label, placeholder):
    screen = Screen(image=SCREEN.image, scale=2.0, app="Google Chrome", field=field(label, placeholder), url=None)
    refusal = actions.perform(SimpleNamespace(chosen=chosen), screen, [], context())
    assert refusal == f"{chosen} refused: the focused field asks for a credential"
    assert typed == []


@pytest.mark.parametrize("label", ["Search", "Shipping address", "Mapping notes", "Spinner count", "Title"])
def test_an_ordinary_field_whose_name_merely_contains_pin_or_key_is_typed_into(typed, label):
    screen = Screen(image=SCREEN.image, scale=2.0, app="Google Chrome", field=field(label), url=None)
    result = actions.perform(SimpleNamespace(chosen="type_text"), screen, [], context())
    assert result.startswith("typed 'quarterly report'") and "quarterly report" in typed
