"""Request compilers: a clear request becomes a URL or a plan step, with no network and no browser."""

from urllib.parse import parse_qs, urlsplit

import pytest

from glide.computer.execution.contracts import Milestone, safe_url
from glide.direct import Direct, DirectConfigError, load_settings, resolve

SETTINGS = load_settings()

# (request, kind, host, check on the whole url)
CLEAR = [
    ("open youtube", "open", "www.youtube.com"),
    ("Open YouTube please", "open", "www.youtube.com"),
    ("go to github", "open", "github.com"),
    ("open the github website", "open", "github.com"),
    ("open https://example.org/docs", "open", "example.org"),
    ("visit www.example.org", "open", "www.example.org"),
    ("search amazon for headphones", "search", "www.amazon.com"),
    ("search for headphones on amazon", "search", "www.amazon.com"),
    ("search youtube for lo-fi beats", "search", "www.youtube.com"),
    ("search for cats", "search", "www.google.com"),
    ("search best pizza near me", "search", "www.google.com"),
    ("directions from Boston to New York", "directions", "www.google.com"),
    ("directions to the airport", "directions", "www.google.com"),
    ("how do I get to Central Park from Times Square", "directions", "www.google.com"),
    ("navigate to Union Station", "directions", "www.google.com"),
    ("show me the Eiffel Tower on the map", "place", "www.google.com"),
    ("map of Lisbon", "place", "www.google.com"),
    ("what's the weather in Paris", "weather", "www.google.com"),
    ("weather in San Francisco", "weather", "www.google.com"),
    ("how is the weather for Zürich", "weather", "www.google.com"),
    ("stock price of AAPL", "quote", "finance.yahoo.com"),
    ("TSLA stock price", "quote", "finance.yahoo.com"),
    ("what is the stock price of apple", "quote", "finance.yahoo.com"),
    ("quote for BRK.B", "quote", "finance.yahoo.com"),
]


@pytest.mark.parametrize(("text", "kind", "host"), CLEAR)
def test_clear_requests_compile(text, kind, host):
    got = resolve(text)
    assert isinstance(got, Direct)
    assert (got.kind, urlsplit(got.url).hostname) == (kind, host)
    assert safe_url(got.url)
    assert got.answer == ""


def test_arithmetic_is_an_answer_not_a_url():
    got = resolve("what is 15% of 240")
    assert got is not None and (got.kind, got.answer, got.url) == ("answer", "36", "")
    assert got.plan_step() is None


def test_query_lands_in_the_query_parameter():
    got = resolve("search amazon for wireless headphones")
    query = parse_qs(urlsplit(got.url).query)
    assert list(query.values()) == [["wireless headphones"]]


def test_directions_carry_both_ends():
    got = resolve("directions from Boston to New York")
    values = sorted(v[0] for v in parse_qs(urlsplit(got.url).query).values())
    assert "Boston" in values and "New York" in values


UNCLEAR = [
    "",
    "open",
    "open the pod bay doors",
    "open notes.txt",
    "open readme.md",
    "open javascript:alert(1)",
    "open file:///etc/passwd",
    "open ftp://example.org/x",
    "open data:text/html,hi",
    "open https://user:pw@example.org/",
    "open http://",
    "search",
    "search for",
    "search ebay for shoes",
    "tell me a joke",
    "what is the weather",
    "weather in",
    "weather in 12345",
    "weather in Paris; rm -rf /",
    "stock price of banana",
    "stock price of",
    "directions",
    "directions to",
    "navigate",
    "map of",
    "send an email to bob",
    "how do I get to sleep",
    "why is the sky blue",
]


@pytest.mark.parametrize("text", UNCLEAR)
def test_unclear_or_unsafe_requests_are_none(text):
    assert resolve(text) is None


INJECTIONS = [
    "x&redirect=https://evil.example/",
    "x#fragment",
    "x?extra=1",
    "../../etc/passwd",
    "x/../../admin",
    "https://evil.example/phish",
    "javascript:alert(1)",
    "<script>alert(1)</script>",
    "a%26b=c",
    "100% sure",
    "x'; DROP TABLE users;--",
    "x{query}{city}",
    "x\\evil.example",
    "x@evil.example",
    "ünïcode 日本語",
    "a+b c",
]


@pytest.mark.parametrize("payload", INJECTIONS)
def test_query_text_stays_data(payload):
    got = resolve(f"search amazon for {payload}")
    assert got is not None
    parts = urlsplit(got.url)
    assert parts.hostname == "www.amazon.com" and parts.scheme == "https"
    assert parts.fragment == "" and parts.username is None
    assert parse_qs(parts.query, keep_blank_values=True) == {"k": [payload]}
    assert safe_url(got.url) and not any(c.isspace() for c in got.url)


@pytest.mark.parametrize("payload", ["x/../../y", "a?b=c#d", "x&y=z", "Mc/Donald"])
def test_path_placeholders_cannot_leave_their_segment(payload):
    got = resolve(f"weather in {payload}")
    if got is None:  # the city grammar may refuse it outright; either way it never becomes a path or a query
        return
    assert urlsplit(got.url).path.count("/") <= urlsplit(SETTINGS.weather).path.count("/")


@pytest.mark.parametrize("payload", ["x\ny", "x\r\nHost: evil.example", "x\x00y", "x" * 500])
def test_control_characters_and_long_queries_are_refused(payload):
    assert resolve(f"search amazon for {payload}") is None
    assert resolve(f"search for {payload}") is None


def test_plan_step_is_a_valid_milestone():
    got = resolve("open youtube")
    step = got.plan_step()
    assert step is not None and step["effect"] == "url" and step["value"] == got.url
    assert set(step) == {"id", "goal", "effect", "target", "value", "quantity"}
    Milestone(**step)


def test_plan_step_goal_does_not_echo_the_query():
    got = resolve("search amazon for secret-thing")
    assert "secret-thing" not in got.plan_step()["goal"]
    assert "secret-thing" not in repr(got)


def test_no_network_or_browser_is_reached(monkeypatch):
    import socket
    import webbrowser

    def boom(*a, **k):
        raise AssertionError("reached out")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(webbrowser, "open", boom)
    assert resolve("search amazon for headphones") is not None
    assert resolve("open youtube") is not None


# --- configuration ---------------------------------------------------------------------------------------


def test_override_replaces_and_adds_aliases_and_templates():
    settings = load_settings(
        {
            "sites": {"Mail": "https://mail.example.test/"},
            "search": {"docs": "https://docs.example.test/find?term={query}", "default": "docs"},
            "weather": "https://weather.example.test/{city}",
        }
    )
    assert resolve("open mail", settings).url == "https://mail.example.test/"
    assert resolve("search docs for a b", settings).url == "https://docs.example.test/find?term=a%20b"
    assert resolve("search for cats", settings).url == "https://docs.example.test/find?term=cats"
    assert resolve("weather in Rome", settings).url == "https://weather.example.test/Rome"
    assert resolve("open youtube", settings) is not None  # defaults kept


@pytest.mark.parametrize(
    "bad",
    [
        {"sites": {"x": "javascript:alert(1)"}},
        {"sites": {"x": "file:///etc/passwd"}},
        {"sites": {"x": "ftp://example.test/"}},
        {"sites": {"x": "//example.test/"}},
        {"sites": {"x": 3}},
        {"search": {"x": "https://example.test/?q="}},  # no {query}
        {"search": {"x": "https://example.test/?q={nope}"}},
        {"search": {"x": "https://{query}.example.test/"}},  # query would choose the host
        {"search": {"x": "https://example.test/{query}", "default": "missing"}},
        {"weather": "https://example.test/{query}"},
        {"weather": "https://example.test/"},
        {"maps": {"place": "https://example.test/{origin}"}},
        {"maps": {"nonsense": "https://example.test/"}},
        {"quote": "https://example.test/{city}"},
        {"tickers": {"apple": "not a ticker!"}},
        {"surprise": {}},
        "not a table",
    ],
)
def test_bad_overrides_are_refused(bad):
    with pytest.raises(DirectConfigError):
        load_settings(bad)


def test_none_and_empty_override_give_the_defaults():
    assert load_settings(None) == load_settings({})


def test_glide_toml_direct_table_round_trips(tmp_path):
    from glide.direct import load_settings_file

    path = tmp_path / "glide.toml"
    path.write_text('[other]\nx = 1\n\n[direct.sites]\nwiki = "https://wiki.example.test/"\n', encoding="utf-8")
    assert resolve("open wiki", load_settings_file(path)).url == "https://wiki.example.test/"
    assert load_settings_file(tmp_path / "missing.toml") == load_settings()


def test_cli_prints_the_compiled_request(capsys):
    from glide.direct.__main__ import main

    assert main(["what is 2 plus 2"]) == 0
    assert capsys.readouterr().out.strip() == "answer: 4"
    assert main(["tell me a joke"]) == 1
