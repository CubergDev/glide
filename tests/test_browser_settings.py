"""Browser provider selection from glide.toml and the environment: explicit, loopback-only, never connecting."""

from __future__ import annotations

import argparse

import pytest

from glide.computer import browser_settings as bs

ENV_NAMES = (
    "GLIDE_BROWSER_PROVIDER",
    "GLIDE_BROWSER_ENDPOINT",
    "GLIDE_BROWSER_TARGET",
    "GLIDE_PLAYWRIGHT_SESSION",
    "GLIDE_PLAYWRIGHT_CLI",
    "GLIDE_SEARCH_URL",
)


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.setenv(name, "")  # so the undo removes whatever apply_arguments sets, even from a test
        monkeypatch.delenv(name)
    monkeypatch.setattr(bs, "_table", {})


def test_nothing_configured_means_the_native_desktop():
    settings = bs.resolve({}, {})
    assert settings.provider == "native" and settings.source == "default" and settings.fallback == ()
    assert settings.connection() == bs.Connection("native", "", "", bs.DEFAULT_SESSION) and settings.search_url == ""


@pytest.mark.parametrize("provider", ["native", "cdp", "obscura", "playwright", "playwright-cli"])
def test_the_provider_the_file_names_is_the_one_used(provider):
    table = {"provider": provider, "cdp": {"endpoint": "http://127.0.0.1:9415"}, "obscura": {"endpoint": "http://localhost:9333"}}
    settings = bs.resolve(table, {})
    assert settings.provider == ("playwright" if provider == "playwright-cli" else provider) and settings.source == "glide.toml"


def test_an_endpoint_without_a_provider_means_cdp_and_environment_beats_the_file():
    assert bs.resolve({"cdp": {"endpoint": "http://127.0.0.1:9222"}}, {}).provider == "cdp"
    assert bs.resolve({}, {"GLIDE_BROWSER_ENDPOINT": "http://127.0.0.1:9222"}).provider == "cdp"
    settings = bs.resolve({"provider": "cdp", "cdp": {"endpoint": "http://127.0.0.1:1"}}, {"GLIDE_BROWSER_PROVIDER": "native"})
    assert settings.provider == "native" and settings.source == "environment"


def test_each_provider_keeps_its_own_endpoint_and_an_override_reaches_only_the_selected_one():
    table = {
        "provider": "obscura",
        "cdp": {"endpoint": "http://127.0.0.1:9222"},
        "obscura": {"endpoint": "http://127.0.0.1:9333"},
    }
    assert bs.resolve(table, {}).connection().endpoint == "http://127.0.0.1:9333"
    assert bs.resolve(table, {}).connection("cdp").endpoint == "http://127.0.0.1:9222"
    settings = bs.resolve(table, {"GLIDE_BROWSER_ENDPOINT": "http://127.0.0.1:9555"})
    assert settings.connection().endpoint == "http://127.0.0.1:9555"
    assert settings.connection("cdp").endpoint == "http://127.0.0.1:9222"


def test_a_stale_endpoint_does_not_constrain_a_provider_that_takes_none():
    for provider in ("native", "playwright"):
        settings = bs.resolve({"provider": provider}, {"GLIDE_BROWSER_ENDPOINT": "https://no-longer-used.test"})
        assert settings.connection().endpoint == ""


def test_playwright_session_and_cli_come_from_the_environment_or_the_file():
    table = {"provider": "playwright", "playwright": {"session": "firefox-work", "cli": "/opt/playwright-cli"}}
    settings = bs.resolve(table, {})
    assert settings.session == "firefox-work" and settings.cli == "/opt/playwright-cli"
    settings = bs.resolve(table, {"GLIDE_PLAYWRIGHT_SESSION": "other", "GLIDE_PLAYWRIGHT_CLI": "/x/cli"})
    assert settings.session == "other" and settings.cli == "/x/cli"
    assert bs.resolve({"provider": "playwright"}, {}).session == bs.DEFAULT_SESSION


@pytest.mark.parametrize(
    "table,env",
    [
        ({"provider": "bogus"}, {}),
        ({"provider": "cdp"}, {}),  # needs an endpoint
        ({"provider": "obscura", "obscura": {"endpoint": "https://remote.test"}}, {}),
        ({"provider": "cdp", "cdp": {"endpoint": "http://example.test:9222"}}, {}),
        ({"provider": "cdp", "cdp": {"endpoint": "http://user:pw@127.0.0.1:9222"}}, {}),
        ({"provider": "cdp", "cdp": {"endpoint": "http://127.0.0.1:9222/json?x=1"}}, {}),
        ({"provider": "cdp", "cdp": {"endpoint": "http://127.0.0.1"}}, {}),  # no port
        ({"provider": "playwright", "playwright": {"session": "-s=bad;echo"}}, {}),
        ({"provider": "playwright"}, {"GLIDE_PLAYWRIGHT_SESSION": "x" * 65}),
        ({"target": "two words"}, {}),
        ({"target": "t" * 257}, {}),
        ({"unknown": 1}, {}),
        ({"cdp": {"endpoint": "http://127.0.0.1:9222", "extra": 1}}, {}),
        ({"provider": 7}, {}),
        ({"cdp": "http://127.0.0.1:9222"}, {}),
    ],
)
def test_a_bad_setting_is_an_error_never_a_quiet_fall_back_to_native(table, env):
    with pytest.raises(ValueError) as caught:
        bs.resolve(table, env)
    for secret in ("pw@", "remote.test", "example.test"):
        assert secret not in str(caught.value)  # the message names the setting, never quotes its value


def test_fallback_is_opt_in_ordered_and_never_the_native_desktop():
    table = {
        "provider": "cdp",
        "fallback": ["obscura", "playwright-cli"],
        "cdp": {"endpoint": "http://127.0.0.1:9222"},
        "obscura": {"endpoint": "http://127.0.0.1:9333"},
    }
    assert bs.resolve(table, {}).chain == ("cdp", "obscura", "playwright")
    assert bs.resolve({k: v for k, v in table.items() if k != "fallback"}, {}).fallback == ()
    for bad in (["native"], ["cdp"], ["obscura", "obscura"], ["obscura-2"], "obscura", [3]):
        with pytest.raises(ValueError):
            bs.resolve({**table, "fallback": bad}, {})
    with pytest.raises(ValueError, match="native desktop has no fallback"):
        bs.resolve({"provider": "native", "fallback": ["playwright"]}, {})
    with pytest.raises(ValueError):  # a fallback provider is validated like the first: an endpoint it cannot use
        bs.resolve({**table, "fallback": ["obscura"], "obscura": {}}, {})


def test_use_installs_the_table_for_the_process_and_rejects_a_bad_one_without_installing_it():
    bs.use({"provider": "cdp", "cdp": {"endpoint": "http://127.0.0.1:9415"}})
    assert bs.current().provider == "cdp" and bs.current().connection().endpoint == "http://127.0.0.1:9415"
    with pytest.raises(ValueError):
        bs.use({"provider": "nope"})
    assert bs.current().provider == "cdp"
    bs.use(None)
    assert bs.current().provider == "native"


def test_the_search_address_comes_from_the_environment_or_the_file_and_there_is_no_built_in_one():
    assert bs.resolve({}, {}).search_url == ""
    table = {"search_url": "https://search.example.test/"}
    assert bs.resolve(table, {}).search_url == "https://search.example.test/"
    assert bs.resolve(table, {"GLIDE_SEARCH_URL": " https://other.example.test "}).search_url == "https://other.example.test"
    assert bs.resolve({}, {"GLIDE_SEARCH_URL": "https://only.example.test"}).search_url == "https://only.example.test"


@pytest.mark.parametrize("value", ["ftp://search.example.test", "search.example.test", "https://user:pw@search.example.test", 7])
def test_a_search_address_that_is_not_a_plain_web_address_is_an_error_that_does_not_quote_it(value):
    for table, env in (({"search_url": value}, {}), ({}, {"GLIDE_SEARCH_URL": str(value)})):
        with pytest.raises(ValueError) as caught:
            bs.resolve(table, env)
        assert "search" in str(caught.value) and "pw@" not in str(caught.value) and "search.example" not in str(caught.value)


def test_description_names_the_provider_and_how_to_reach_it():
    bs.use({"provider": "obscura", "obscura": {"endpoint": "http://127.0.0.1:9333"}, "target": "new"})
    assert bs.description() == "Obscura · http://127.0.0.1:9333 · target new"
    bs.use({"provider": "playwright", "playwright": {"session": "s1"}})
    assert bs.description() == "Playwright CLI · session s1 · target single tab"


def test_command_line_choices_become_the_environment_and_are_checked(monkeypatch):
    parser = argparse.ArgumentParser()
    bs.add_arguments(parser)
    bs.apply_arguments(parser.parse_args(["--browser-provider", "playwright-cli", "--playwright-session", "fixture"]))
    assert bs.current().connection() == bs.Connection("playwright", "", "", "fixture")
    with pytest.raises(ValueError):
        bs.apply_arguments(parser.parse_args(["--browser-provider", "cdp", "--browser-endpoint", "https://remote.test"]))
