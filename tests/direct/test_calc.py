"""The arithmetic evaluator: exact answers for clear arithmetic, None for everything else."""

import pytest

from glide.direct.calc import evaluate

ANSWERS = [
    ("what is 15% of 240", "36"),
    ("15 percent of 240", "36"),
    ("twelve times eleven", "132"),
    ("what's twelve times eleven?", "132"),
    ("3.5 plus 4 over 2", "5.5"),
    ("2 + 2", "4"),
    ("2+2", "4"),
    ("10 - 3 * 2", "4"),
    ("(10 - 3) * 2", "14"),
    ("7 x 6", "42"),
    ("100 divided by 8", "12.5"),
    ("1 / 3", "0.3333333333"),
    ("0.1 + 0.2", "0.3"),
    ("2 ^ 10", "1024"),
    ("2 ** 3 ** 2", "512"),
    ("two to the power of eight", "256"),
    ("9 squared", "81"),
    ("3 cubed", "27"),
    ("square root of 144", "12"),
    ("what is minus 5 plus 8", "3"),
    ("-5 + 8", "3"),
    ("17 mod 5", "2"),
    ("one hundred and five minus five", "100"),
    ("twenty one plus thirty four", "55"),
    ("two thousand five hundred divided by five", "500"),
    ("three point five times two", "7"),
    ("1,000 plus 250", "1250"),
    ("$5 plus $7", "12"),
    ("15 dollars plus 5 dollars", "20"),
    ("3 km times 2", "6"),
    ("how much is 8 times 7", "56"),
    ("calculate 9 minus 12", "-3"),
    ("what is 6 times 7 please", "42"),
    ("12 times 12 equals", "144"),
    ("What Is 5 Plus 5", "10"),
]


@pytest.mark.parametrize(("text", "expected"), ANSWERS)
def test_clear_arithmetic_is_answered_exactly(text, expected):
    assert evaluate(text) == expected


NOT_ARITHMETIC = [
    "",
    "42",
    "hello there",
    "what is the capital of France",
    "I need three plus two more things from the shop",
    "open youtube",
    "call 555-1234",
    "2024-05-06",
    "12/05/2024",
    "what is 5 people",
    "five plus",
    "plus five",
    "times",
    "3 +",
    "(3 + 4",
    "3 + 4)",
    "what is 5 percent",
    "50%",
    "5 % 3",
    "__import__('os').system('ls')",
    "2 + __import__('os')",
    "open('/etc/passwd')",
    "1 if 1 else 2",
    "3 plus four apples",
    "I waited 2 times 3 hours",
    "one",
    "ten ten",
    "five six plus two",
    "1 / 0",
    "5 mod 0",
    "square root of minus 4",
    "10 to the power of 0.5",
]


@pytest.mark.parametrize("text", NOT_ARITHMETIC)
def test_anything_else_is_none(text):
    assert evaluate(text) is None


BOUNDS = [
    "2 ^ 1000",
    "9 ^ 9 ^ 9",
    "2 ** 99999999",
    "10 ^ 65",
    "1" + "0" * 40 + " + 1",
    "99999999999999999999 times 99999999999999999999 times 99999999999999999999",
    " + ".join(["1"] * 200),
    "1 + 1" + " " * 500,
    "(" * 100 + "1" + ")" * 100 + " + 1",
]


@pytest.mark.parametrize("text", BOUNDS)
def test_size_and_exponent_are_bounded(text):
    assert evaluate(text) is None


def test_small_exponents_still_work():
    assert evaluate("2 ^ 64") == "18446744073709551616"
    assert evaluate("2 ^ -2") == "0.25"


def test_no_eval_or_exec_in_the_module():
    import ast
    from pathlib import Path

    import glide.direct.calc as calc

    tree = ast.parse(Path(calc.__file__).read_text(encoding="utf-8"))
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"eval", "exec", "compile", "__import__"}
