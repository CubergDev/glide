import os

import pytest

from glide.computer.config import load_dotenv
from glide.computer.writer import WriterError, parse_json, valid_url


def test_dotenv_sets_only_missing_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("GLIDE_TEST_PRESENT", "keep")
    monkeypatch.setenv("GLIDE_TEST_NEW", "")  # so the undo removes what load_dotenv sets, and no later test sees it
    monkeypatch.delenv("GLIDE_TEST_NEW")
    (tmp_path / ".env").write_text('# comment\nGLIDE_TEST_PRESENT=override\nGLIDE_TEST_NEW="quoted value"\nbroken line\n')
    load_dotenv(tmp_path / ".env")
    assert os.environ["GLIDE_TEST_PRESENT"] == "keep"
    assert os.environ["GLIDE_TEST_NEW"] == "quoted value"


def test_dotenv_missing_file_is_fine(tmp_path):
    load_dotenv(tmp_path / "nope.env")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"fill": true}', {"fill": True}),
        ('```json\n{"fill": false}\n```', {"fill": False}),
    ],
)
def test_the_writer_reply_is_read_plain_or_in_a_code_fence(reply, expected):
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
