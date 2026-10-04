"""The safety policy: a pure table from a proposed UI action to allow, ask, refuse or handoff.

Rows are (action fields, decision, reason). Nothing here reaches the machine, so no guard row is needed.
"""

# ruff: noqa: RUF001  look-alike letters are the point of these rows

from __future__ import annotations

import json

import pytest

from glide.computer.writer import CREDENTIAL_HINTS
from glide.safety_policy import Action, Decision, Lexicon, Reason, decide, load_lexicon, normalize
from glide.safety_policy.__main__ import main

A, ASK, REFUSE, HAND = Decision.ALLOW, Decision.ASK, Decision.REFUSE, Decision.HANDOFF

CLICKS = [
    # (label, role, decision, reason)
    ("Place order", "AXButton", ASK, Reason.SPENDING),
    ("PLACE   ORDER", "AXButton", ASK, Reason.SPENDING),
    ("  place\torder  ", "button", ASK, Reason.SPENDING),
    ("pla\u200bce order", "button", ASK, Reason.SPENDING),  # zero-width space inside a word
    ("pla\u00adce order", "button", ASK, Reason.SPENDING),  # soft hyphen
    ("Рlасе оrdеr", "button", ASK, Reason.SPENDING),  # Cyrillic look-alikes
    ("ｐｌａｃｅ ｏｒｄｅｒ", "button", ASK, Reason.SPENDING),  # full width
    ("placeOrder", "button", ASK, Reason.SPENDING),
    ("place_order", "button", ASK, Reason.SPENDING),
    ("Pay-now", "Hyperlink", ASK, Reason.SPENDING),
    ("Buy", "button", ASK, Reason.SPENDING),
    ("Subscribe", "button", ASK, Reason.SPENDING),
    ("Send", "AXButton", ASK, Reason.COMMUNICATION),
    ("S e n d", "AXButton", ASK, Reason.COMMUNICATION),  # letters spread out
    ("Reply all", "button", ASK, Reason.COMMUNICATION),
    ("Post", "button", ASK, Reason.COMMUNICATION),
    ("Submit", "button", ASK, Reason.COMMUNICATION),
    ("Delete", "AXButton", ASK, Reason.DESTRUCTIVE),
    ("DELETE", "link", ASK, Reason.DESTRUCTIVE),
    ("Remove from cart", "button", ASK, Reason.DESTRUCTIVE),
    ("Cancel subscription", "button", ASK, Reason.DESTRUCTIVE),
    ("Empty trash", "AXMenuItem", ASK, Reason.DESTRUCTIVE),
    ("Save changes", "button", ASK, Reason.OVERWRITE),
    ("Overwrite", "button", ASK, Reason.OVERWRITE),
    ("Accept all cookies", "button", ASK, Reason.CONSENT),
    ("I agree", "checkbox", ASK, Reason.CONSENT),
    # Whole words only: these must not trigger.
    ("Sendero", "button", A, None),
    ("Buyer's guide", "link", A, None),
    ("Posts", "tab", A, None),
    ("Payroll", "link", A, None),
    ("Resend code", "link", A, None),
    ("Deleted items", "link", A, None),
    ("Checkouts history", "link", A, None),
    ("Unsubscribe", "link", A, None),  # its own word, and undoing a subscription is not spending
    ("unsubscribe-nothing", "link", A, None),
    ("Cancel", "button", A, None),
    ("Next", "button", A, None),
    ("Forgot password?", "link", A, None),  # a link to a reset page is not a secret field
    ("Show password", "button", A, None),
    ("Search", "AXButton", A, None),
    # Unlabeled or unknown role: ask, not allow.
    ("", "button", ASK, Reason.UNLABELED),
    ("   ", "button", ASK, Reason.UNLABELED),
    ("\u200b\u200d", "AXButton", ASK, Reason.UNLABELED),
    ("Next", "", ASK, Reason.UNKNOWN),
    ("Next", "AXWeirdThing", ASK, Reason.UNKNOWN),
    ("Delete", "AXWeirdThing", ASK, Reason.DESTRUCTIVE),  # the specific reason wins over the generic one
    # Roles real callers pass.
    ("Next", "AXLink", A, None),
    ("Next", "TabItem", A, None),
    ("Next", "menuitemcheckbox", A, None),
    ("Next", "AXPopUpButton", A, None),
]


@pytest.mark.parametrize(("label", "role", "decision", "reason"), CLICKS)
def test_click_table(label, role, decision, reason):
    verdict = decide(Action("click", label=label, role=role))
    assert (verdict.decision, verdict.reason) == (decision, reason)


def test_label_split_across_fields_is_still_one_phrase():
    for fields in (
        {"label": "Place", "hint": "order"},
        {"label": "Confirm", "value": "payment"},
        {"label": "", "hint": "Delete"},  # a tooltip names the control
        {"label": "Go", "role": "AXButton Delete"},
    ):
        assert decide(Action("click", **{"role": "button", **fields})).decision is ASK, fields


def test_input_value_is_not_scanned_but_a_button_value_is():
    assert decide(Action("click", label="Notes", role="textbox", value="please delete this")).decision is A
    assert decide(Action("click", label="", role="button", value="Delete")).reason is Reason.DESTRUCTIVE


SECRET_FIELDS = [
    {"label": "Password", "role": "textbox"},
    {"label": "Passwords", "role": "textbox"},
    {"label": "PASS WORD", "role": "textbox", "field_type": "password"},
    {"label": "", "role": "AXSecureTextField"},
    {"label": "Name", "role": "textbox", "field_type": "password"},
    {"label": "Name", "role": "textbox", "secret": True},
    {"label": "", "role": "textbox", "field_name": "cc-number"},
    {"label": "", "role": "textbox", "field_name": "cardNumber"},
    {"label": "Card number", "role": "textbox", "field_type": "tel"},
    {"label": "Security code", "role": "AXTextField"},
    {"label": "Expiry date", "role": "AXTextField"},
    {"label": "Pаsswоrd", "role": "AXTextField"},  # Cyrillic look-alikes
    {"label": "SSN", "role": "Edit"},
    {"label": "Passport number", "role": "Edit"},
    {"label": "One-time code", "role": "textbox"},
    {"label": "Enter your PIN", "role": "textbox"},
]


@pytest.mark.parametrize("fields", SECRET_FIELDS)
@pytest.mark.parametrize("kind", ["type", "click", "key"])
def test_secret_field_is_refused_for_every_kind(fields, kind):
    verdict = decide(Action(kind, text="hunter2" if kind == "type" else "", key="return" if kind == "key" else "", **fields))
    assert (verdict.decision, verdict.reason) == (REFUSE, Reason.SECRET_FIELD)


def test_secret_is_refused_even_when_the_task_text_holds_it():
    verdict = decide(Action("type", label="Password", role="textbox", text="hunter2", task_text="my password is hunter2"))
    assert verdict.decision is REFUSE


@pytest.mark.parametrize("label", ["Shipping method", "Spinner", "Pinterest board", "Tokyo office", "Secretary", "Expiryless"])
def test_secret_words_do_not_match_inside_other_words(label):
    verdict = decide(Action("type", label=label, role="textbox", text="x", task_text="x"))
    assert verdict.decision is A


@pytest.mark.parametrize("hint", CREDENTIAL_HINTS)
def test_every_existing_credential_hint_is_refused_as_a_field_label(hint):
    for label in (hint, hint + "s", hint.replace(" ", "-"), hint.upper(), f"Your {hint} here"):
        verdict = decide(Action("type", label=label, role="AXTextField", text="x", task_text="x"))
        assert verdict.decision is REFUSE, label


def test_clicking_a_non_input_with_a_secret_word_is_not_refused():
    assert decide(Action("click", label="Forgot your password", role="link")).decision is A


TYPES = [
    # (label, role, text, task_text, supplied, decision, reason)
    ("Shipping address", "textbox", "12 Elm St", "", (), HAND, Reason.PERSONAL),
    ("Shipping address", "textbox", "12 Elm St", "ship it to 12 Elm St please", (), A, None),
    ("Shipping address", "textbox", "12 elm   st", "ship it to 12 Elm St", (), A, None),
    ("Email", "textbox", "me@example.com", "", ("me@example.com",), A, None),  # configured value
    ("Email", "textbox", "me@example.com", "mail me@example.com", (), A, None),
    ("Email", "textbox", "me@example.com", "mail someone", (), HAND, Reason.PERSONAL),
    ("First name", "AXTextField", "ann", "planning a trip", (), HAND, Reason.PERSONAL),  # not a substring match
    ("First name", "AXTextField", "Ann", "call me Ann.", (), A, None),
    ("First name", "AXTextField", "Аnn", "call me Ann", (), HAND, Reason.PERSONAL),  # look-alike is not supplied
    ("Phone", "Edit", "555 0100", "", (), HAND, Reason.PERSONAL),
    ("Date of birth", "textbox", "1990-01-01", "", (), HAND, Reason.PERSONAL),
    ("emailAddress", "textbox", "x@y.z", "", (), HAND, Reason.PERSONAL),
    ("Search by name", "searchbox", "pizza", "", (), A, None),
    ("File name", "textbox", "report", "", (), A, None),
    ("Spinner", "textbox", "x", "", (), A, None),
    ("Message", "textbox", "hello", "", (), A, None),  # typing is not sending
    ("City", "textbox", "", "", (), A, None),  # clearing invents nothing
    ("", "textbox", "x", "", (), ASK, Reason.UNLABELED),
    ("", "textbox", "x", "type x", (), A, None),
    ("Notes", "AXWeirdThing", "x", "x", (), A, None),  # typing does not depend on the role table
]


@pytest.mark.parametrize(("label", "role", "text", "task", "supplied", "decision", "reason"), TYPES)
def test_type_table(label, role, text, task, supplied, decision, reason):
    verdict = decide(Action("type", label=label, role=role, text=text, task_text=task, supplied=supplied))
    assert (verdict.decision, verdict.reason) == (decision, reason)


@pytest.mark.parametrize("newline", ["\n", "\r", "\r\n", "\u2028"])
def test_a_typed_line_break_is_a_return(newline):
    base = {"kind": "type", "label": "Message", "role": "textbox", "text": "hi" + newline}
    assert (decide(Action(**base)).decision, decide(Action(**base)).reason) == (ASK, Reason.COMMUNICATION)
    assert decide(Action(**base, send_intent=True)).decision is A
    assert decide(Action("type", label="", role="textbox", text="hi\n", task_text="hi")).reason is Reason.UNLABELED


KEYS = [
    # (key, label, role, send_intent, decision, reason)
    ("Enter", "Message", "textbox", False, ASK, Reason.COMMUNICATION),
    ("return", "Write a comment", "AXTextArea", False, ASK, Reason.COMMUNICATION),
    ("cmd+Enter", "Message", "textbox", False, ASK, Reason.COMMUNICATION),
    ("Ctrl+Return", "Chat", "textbox", False, ASK, Reason.COMMUNICATION),
    ("NumpadEnter", "Message", "textbox", False, ASK, Reason.COMMUNICATION),
    ("Enter", "Message", "textbox", True, A, None),
    ("Enter", "Search", "searchbox", False, A, None),
    ("Enter", "Search messages", "searchbox", False, A, None),  # a search field is not a message field
    ("Enter", "City", "textbox", False, A, None),
    ("Enter", "", "textbox", False, ASK, Reason.UNLABELED),
    ("Enter", "City", "", False, ASK, Reason.UNKNOWN),
    ("Enter", "Delete", "button", False, ASK, Reason.DESTRUCTIVE),
    ("Enter", "Pay now", "AXButton", True, ASK, Reason.SPENDING),  # send intent does not cover spending
    ("Escape", "", "", False, A, None),
    ("Tab", "", "", False, A, None),
    ("a", "", "textbox", False, A, None),
]


@pytest.mark.parametrize(("key", "label", "role", "intent", "decision", "reason"), KEYS)
def test_key_table(key, label, role, intent, decision, reason):
    verdict = decide(Action("key", key=key, label=label, role=role, send_intent=intent))
    assert (verdict.decision, verdict.reason) == (decision, reason)


@pytest.mark.parametrize("kind", ["scroll", "hover", "wait", "inspect", "screenshot"])
def test_passive_kinds_are_allowed_even_unlabeled(kind):
    assert decide(Action(kind)).decision is A


def test_an_unknown_kind_is_asked_about():
    assert decide(Action("drag", label="Next", role="button")).reason is Reason.UNKNOWN


def test_verdict_never_echoes_the_label_value_or_typed_text():
    action = Action("type", label="Secret handshake Shipping address", role="textbox", text="ZZTOP-12", value="VVV-9")
    verdict = decide(action)
    shown = repr(verdict) + json.dumps(verdict.as_dict())
    assert "ZZTOP" not in shown and "VVV" not in shown and "handshake" not in shown


def test_normalize_folds_case_accents_lookalikes_and_separators():
    assert normalize("Ｐlace-ORDER") == ("place", "order")
    assert normalize("Cаfé\u200b_Menu") == ("cafe", "menu")
    assert normalize("p a s s w o r d") == ("p", "a", "s", "s", "w", "o", "r", "d")
    assert normalize("") == ()


def test_more_languages_are_added_by_data():
    lexicon = load_lexicon(extra={"lang": {"es": {"communication": ["enviar"], "spending": ["comprar ahora"]}}})
    assert decide(Action("click", label="Enviar", role="button"), lexicon).reason is Reason.COMMUNICATION
    assert decide(Action("click", label="Comprar  ahora", role="button"), lexicon).reason is Reason.SPENDING
    assert decide(Action("click", label="Enviarlo", role="button"), lexicon).decision is A
    assert decide(Action("click", label="Enviar", role="button")).decision is A  # the default lexicon is untouched
    assert isinstance(lexicon, Lexicon)


def test_cli_prints_the_decision_only(capsys):
    code = main(["--kind", "click", "--role", "button", "--label", "Place order SECRETLABEL"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out == {"decision": "ask", "reason": "spending", "term": "place order"}
    assert main(["--kind", "type", "--role", "textbox", "--label", "Password", "--text", "abc"]) == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "refuse"
