"""The fixed Playwright driver script, checked as text and against the Python side that feeds it.

Its behavior (the cancellation gate, key and button release after Stop, the unresolved-operation check) is JavaScript
and runs inside the user's `playwright-cli`; nothing offline can execute it, and the suite does not use Node.
`docs/live-checks/browsers.md` lists the manual checks, including an offline Node harness for the driver.
What can be proved here: the request the driver reads is the one Python writes, every method Python may send has a
handler and none else has, and no model or page text is ever part of the program.
"""

import json
import re

from glide.computer.execution import playwright_cli

DRIVER = playwright_cli.DRIVER.read_text()
HANDLED = set(re.findall(r"request\.method===['\"]([A-Za-z.]+)['\"]", DRIVER))


def test_the_driver_has_exactly_one_placeholder_and_it_is_the_one_python_fills():
    assert re.findall(r"__[A-Z_]+__", DRIVER) == ["__GLIDE_REQUEST__"]
    assert playwright_cli.DRIVER.name == "playwright_driver.js"
    assert not re.search(r"__permit|Permit\.|permit-", DRIVER)  # renamed to glide (D12)


def test_every_method_python_may_send_has_a_handler_and_the_driver_handles_nothing_else():
    assert playwright_cli.METHODS - {"Target.detachFromTarget"} == HANDLED  # detach is answered in Python, never sent


def test_the_driver_reads_only_what_python_writes():
    request = {"method": "Page.navigate", "params": {"url": "u"}, "target": "t", "deadline": 1.0, "gate": "g"}
    for key in set(re.findall(r"request\.([a-z]+)\b", DRIVER)) - {"method", "params", "target"}:
        assert key in request, f"the driver reads request.{key}, which Python does not send"


def test_hostile_text_stays_data_when_the_command_file_is_built(monkeypatch):
    hostile = "'; process.exit(4); //\n`${require('child_process')}` __GLIDE_REQUEST__"
    request = {"method": "Input.insertText", "params": {"text": hostile}, "target": "t", "deadline": 1.0, "gate": "g"}
    code = DRIVER.replace("__GLIDE_REQUEST__", json.dumps(request, ensure_ascii=True), 1)
    literal = code.split("const request = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(literal) == request
    assert code.count("process.exit") == 1  # only inside the JSON string literal
