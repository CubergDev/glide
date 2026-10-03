import os

import pytest

from glide.computer.config import custom_writer_endpoint, load_dotenv, writer_base_url
from glide.computer.writer import WriterError, parse_json, valid_url


def test_dotenv_sets_only_missing_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("CLICKER_TEST_PRESENT", "keep")
    monkeypatch.delenv("CLICKER_TEST_NEW", raising=False)
    (tmp_path / ".env").write_text('# comment\nCLICKER_TEST_PRESENT=override\nCLICKER_TEST_NEW="quoted value"\nbroken line\n')
    load_dotenv(tmp_path / ".env")
    assert os.environ["CLICKER_TEST_PRESENT"] == "keep"
    assert os.environ["CLICKER_TEST_NEW"] == "quoted value"


def test_dotenv_missing_file_is_fine(tmp_path):
    load_dotenv(tmp_path / "nope.env")


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("http://localhost:8081", "http://localhost:8081"),
        ("http://localhost:8081/", "http://localhost:8081"),
        ("http://localhost:8081/v1", "http://localhost:8081"),
        ("http://localhost:8081/v1/messages", "http://localhost:8081"),
        ("https://proxy.example.com/v1/messages/", "https://proxy.example.com"),
    ],
)
def test_the_writer_endpoint_accepts_any_of_its_spellings(given, expected, clean_env):
    clean_env.setenv("CLICKER_WRITER_BASE_URL", given)
    assert writer_base_url() == expected


def test_the_anthropic_base_url_is_left_to_the_sdk(clean_env):
    clean_env.setenv("ANTHROPIC_BASE_URL", "http://localhost:8081")
    assert writer_base_url() is None


@pytest.mark.parametrize(
    ("env", "custom"),
    [
        ({}, False),
        ({"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}, False),
        ({"ANTHROPIC_BASE_URL": "http://localhost:1234"}, True),
        ({"CLICKER_WRITER_BASE_URL": "http://localhost:1234/v1/messages"}, True),
    ],
)
def test_the_endpoint_is_custom_whichever_variable_names_it(clean_env, env, custom):
    for name, value in env.items():
        clean_env.setenv(name, value)
    assert custom_writer_endpoint() is custom


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"fill": true}', {"fill": True}),
        ('```json\n{"fill": false}\n```', {"fill": False}),
        ('Sure thing:\n{"fill": true, "text": "hi"}\nHope that helps.', {"fill": True, "text": "hi"}),
    ],
)
def test_the_writer_reply_is_read_whether_or_not_it_came_plain(reply, expected):
    assert parse_json(reply) == expected


def test_a_reply_without_any_json_says_so():
    with pytest.raises(WriterError, match="without usable JSON"):
        parse_json("I am not able to help with that.")


def test_valid_url():
    assert valid_url("https://www.cnn.com")
    assert valid_url("https://news.ycombinator.com/newest")
    assert not valid_url("http://www.cnn.com")
    assert not valid_url("https://localhost")
    assert not valid_url("https://www.cnn.com/a b")
    assert not valid_url("")
