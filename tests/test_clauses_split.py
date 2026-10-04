"""The clause splitter: ordered steps, what each waits for, quotes and names left whole."""

from __future__ import annotations

import pytest

from glide.clauses import Plan, Step, split_clauses


def shape(text):
    return [(s.text, s.depends_on) for s in split_clauses(text).steps]


CASES = [
    # one request stays one step
    ("open notes", [("open notes", ())]),
    ("", []),
    ("   ", []),
    # sequential markers
    ("open notes then close it", [("open notes", ()), ("close it", (0,))]),
    ("search for cats, then open the first result", [("search for cats", ()), ("open the first result", (0,))]),
    ("open notes and then write hello", [("open notes", ()), ("write hello", (0,))]),
    ("open mail. after that open notes", [("open mail", ()), ("open notes", (0,))]),
    ("open mail afterwards open notes", [("open mail", ()), ("open notes", (0,))]),
    ("play music once done close it", [("play music", ()), ("close it", (0,))]),
    ("play music once that's done open mail", [("play music", ()), ("open mail", (0,))]),
    # a comma between two actions is an order
    ("search for cats, open the first result", [("search for cats", ()), ("open the first result", (0,))]),
    # "and" before an action that needs the earlier result waits for it
    ("open notes and write hello", [("open notes", ()), ("write hello", (0,))]),
    ("open notes and write hello then close it", [("open notes", ()), ("write hello", (0,)), ("close it", (1,))]),
    # "and" joining independent actions may run together
    ("open notes and open calculator", [("open notes", ()), ("open calculator", ())]),
    (
        "open notes and open calculator then close both",
        [("open notes", ()), ("open calculator", ()), ("close both", (0, 1))],
    ),
    # a pronoun points at the step before, even after a parallel "and"
    ("open mail and play it", [("open mail", ()), ("play it", (0,))]),
    # names, numbers, lists and single verbs are not split
    ("play Tom and Jerry", [("play Tom and Jerry", ())]),
    ("open notes and calculator", [("open notes and calculator", ())]),
    ("set a timer for 1,000 seconds", [("set a timer for 1,000 seconds", ())]),
    ("copy and paste", [("copy and paste", ())]),
    ("search for rock and roll", [("search for rock and roll", ())]),
    ("ask Dr. Smith to open notes", [("ask Dr. Smith to open notes", ())]),
    ("it is 3.5 degrees", [("it is 3.5 degrees", ())]),
    # dangling markers vanish
    ("then open notes", [("open notes", ())]),
    ("open notes then", [("open notes", ())]),
    ("open notes, and", [("open notes", ())]),
    # mishearings of the join words
    ("open notes and than close it", [("open notes", ()), ("close it", (0,))]),
    ("open notes, than close it", [("open notes", ()), ("close it", (0,))]),
    ("open notes an then close it", [("open notes", ()), ("close it", (0,))]),
    ("open notes afterthat close it", [("open notes", ()), ("close it", (0,))]),
    ("OPEN notes THEN close it", [("OPEN notes", ()), ("close it", (0,))]),
    ("a better offer than this", [("a better offer than this", ())]),
]


@pytest.mark.parametrize(("text", "want"), CASES)
def test_split(text, want):
    assert shape(text) == want


QUOTES = [
    ('type "hello and then goodbye" then close it', [('type "hello and then goodbye"', ()), ("close it", (0,))]),
    ("type 'one, two and open three' and close it", [("type 'one, two and open three'", ()), ("close it", (0,))]),
    ("write “first. then second”", [("write “first. then second”", ())]),
    ('say "don\'t stop" and open mail', [('say "don\'t stop"', ()), ("open mail", ())]),
    ("don't open notes and then it's fine", [("don't open notes", ()), ("it's fine", (0,))]),
    ('type "unclosed then open mail', [('type "unclosed then open mail', ())]),
]


@pytest.mark.parametrize(("text", "want"), QUOTES)
def test_quotes_are_never_split(text, want):
    assert shape(text) == want


def test_steps_are_numbered_and_depend_only_backwards():
    plan = split_clauses("open a and open b then close it, then quit")
    assert isinstance(plan, Plan) and all(isinstance(s, Step) for s in plan.steps)
    assert [s.index for s in plan.steps] == list(range(len(plan.steps)))
    assert all(d < s.index for s in plan.steps for d in s.depends_on)


def test_refers_back_marks_a_pronoun_step():
    plan = split_clauses("open notes then close it")
    assert [s.refers_back for s in plan.steps] == [False, True]
    assert split_clauses("close it").steps[0].depends_on == ()  # nothing before it


def test_parallel_flag_names_steps_that_may_run_together():
    plan = split_clauses("open notes and open calculator then quit")
    assert plan.steps[0].depends_on == plan.steps[1].depends_on == ()
    assert plan.parallel_groups() == [(0, 1), (2,)]


def test_same_input_same_answer():
    text = "open notes and open mail then close it"
    assert split_clauses(text) == split_clauses(text)
