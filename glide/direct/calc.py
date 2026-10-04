"""Arithmetic spoken or typed, answered without a model: `evaluate("what is 15% of 240") -> "36"`.

The text is turned into tokens, parsed into a small tree of our own (numbers, negation, the four operations, modulo,
percent-of, power and square root) and the tree is evaluated in `Decimal`. Nothing is ever handed to `eval`, `exec` or
`ast`, so there is nothing to inject into. It answers only when the whole text is arithmetic: one unknown word, a stray
bracket, a date or a phone number, or a lone number with no operation, and the answer is None, so an ordinary
sentence is never captured and the caller carries on to its router. It is bounded in text length, tokens, nesting,
digits, exponent and result size. A division by zero or an undefined result is also None (the caller's model explains it).
"""

# ruff: noqa: RUF001  the multiplication and division signs are real input

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, localcontext

MAX_CHARS = 200
MAX_TOKENS = 60
MAX_DEPTH = 12
MAX_DIGITS = 30  # of a literal, and of the integer part of any result
MAX_EXPONENT = 100
PLACES = 10

_LEAD = re.compile(
    r"^(?:(?:hey |ok )?glide[, ]+)?(?:please )?(?:(?:what(?:'s| is)|whats|how much is|calculate|compute|work out|tell me)\s+)"
)
_TAIL = re.compile(r"(?:\s+please|\s+equals?|\s*=)+$")
_PHRASES = (  # spoken operations that take several words, or follow their operand
    (r"%\s*of\b|\bper ?cent of\b", " %of "),
    (r"\bto the power of\b|\bto the power\b|\braised to\b", " ^ "),
    (r"\bmultiplied by\b|\bmultiply by\b", " * "),
    (r"\bdivided by\b", " / "),
    (r"\bsquare root of\b", " sqrt "),
    (r"\bsquared\b", " ^ 2 "),
    (r"\bcubed\b", " ^ 3 "),
)
_WORD_OPS = {"plus": "+", "add": "+", "minus": "-", "negative": "-", "subtract": "-", "times": "*", "x": "*", "over": "/"}
_WORD_OPS |= {"mod": "mod", "modulo": "mod"}
_TOKEN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\*\*|[-+*/^()×÷]|%of|[a-z]+")
_UNITS = frozenset(
    [
        "dollar",
        "buck",
        "euro",
        "pound",
        "km",
        "kilometer",
        "kilometre",
        "mile",
        "meter",
        "metre",
        "cm",
        "mm",
        "kg",
        "gram",
        "lb",
        "liter",
        "litre",
        "ml",
        "minute",
        "hour",
        "second",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "degree",
        "mph",
    ]
)

_ONES = {w: i for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])}
_TEENS = {
    w: 10 + i
    for i, w in enumerate(
        ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    )
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_NUMBER_WORDS = {*_ONES, *_TEENS, *_TENS, *_SCALES, "hundred", "and", "point"}


@dataclass(frozen=True)
class Num:
    value: Decimal


@dataclass(frozen=True)
class Neg:
    operand: Node


@dataclass(frozen=True)
class Sqrt:
    operand: Node


@dataclass(frozen=True)
class Bin:
    op: str
    left: Node
    right: Node


Node = Num | Neg | Sqrt | Bin


class NotArithmetic(Exception):
    """The text is not (clearly, safely) arithmetic. Internal: `evaluate` turns it into None."""


def _words_to_number(words: list[str]) -> Decimal:
    """A run of number words ("one hundred and five", "three point five") as a number, or NotArithmetic."""
    if "point" in words:
        cut = words.index("point")
        whole, fraction = words[:cut], words[cut + 1 :]
        if not fraction or any(w not in _ONES for w in fraction) or "point" in fraction:
            raise NotArithmetic
        tail = Decimal("0." + "".join(str(_ONES[w]) for w in fraction))
        return (_words_to_number(whole) if whole else Decimal(0)) + tail
    total, current, last_scale, previous = 0, 0, 10**12, ""
    for i, word in enumerate(words):
        if word == "and":
            if previous not in {"hundred", *_SCALES} or i + 1 == len(words):
                raise NotArithmetic
        elif word == "zero":
            if len(words) != 1:
                raise NotArithmetic
        elif word in _ONES:
            if current % 100 not in {0, 20, 30, 40, 50, 60, 70, 80, 90}:
                raise NotArithmetic
            current += _ONES[word]
        elif word in _TEENS or word in _TENS:
            if current % 100:
                raise NotArithmetic
            current += _TEENS.get(word) or _TENS[word]
        elif word == "hundred":
            if not 1 <= current <= 9:
                raise NotArithmetic
            current *= 100
        else:  # a scale
            if current == 0 or _SCALES[word] >= last_scale:
                raise NotArithmetic
            total, current, last_scale = total + current * _SCALES[word], 0, _SCALES[word]
        previous = word
    if previous == "and":
        raise NotArithmetic
    return Decimal(total + current)


def _tokens(text: str) -> list[str | Decimal]:
    """Operators as their symbol, numbers as Decimal. Units after a number are dropped; any other word refuses."""
    lead = _LEAD.search(text)
    if lead:
        text = text[lead.end() :]
    text = _TAIL.sub("", text.replace("$", "").replace("€", "").replace("£", ""))
    if not lead and re.search(r"\d[-/]\d", text):
        raise NotArithmetic  # a date or a phone number
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)
    for pattern, replacement in _PHRASES:
        text = re.sub(pattern, replacement, text)
    found = _TOKEN.findall(text)
    if re.sub(r"\s+", "", "".join(found)) != re.sub(r"\s+", "", text):
        raise NotArithmetic  # a character that is not part of arithmetic
    if len(found) > MAX_TOKENS:
        raise NotArithmetic
    out: list[str | Decimal] = []
    i = 0
    while i < len(found):
        token = found[i]
        if token[0].isdigit():
            if len(token.replace(",", "").split(".")[0]) > MAX_DIGITS:
                raise NotArithmetic
            out.append(Decimal(token.replace(",", "")))
        elif token in _WORD_OPS:
            out.append(_WORD_OPS[token])
        elif token in _NUMBER_WORDS:
            j = i
            while j < len(found) and found[j] in _NUMBER_WORDS:
                j += 1
            out.append(_words_to_number(list(found[i:j])))
            i = j
            continue
        elif token in {"sqrt", "%of", "**"} or not token[0].isalpha():
            out.append({"×": "*", "÷": "/", "**": "^"}.get(token, token))
        elif (token.removesuffix("s") in _UNITS or token in _UNITS) and out and isinstance(out[-1], Decimal):
            pass  # "15 dollars": the unit is ignored
        else:
            raise NotArithmetic
        i += 1
    return out


class _Parser:
    """expr: term (+|- term)*; term: unary (*|/|mod|%of unary)*; unary: -unary | power; power: atom (^ unary)?"""

    def __init__(self, tokens: list[str | Decimal]):
        self.tokens, self.at, self.depth, self.operations = tokens, 0, 0, 0

    def peek(self) -> str | Decimal | None:
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def take(self) -> str | Decimal:
        token = self.peek()
        if token is None:
            raise NotArithmetic
        self.at += 1
        return token

    def parse(self) -> Node:
        tree = self.expr()
        if self.peek() is not None or not self.operations:
            raise NotArithmetic  # trailing text, or a number with no operation on it
        return tree

    def expr(self) -> Node:
        node = self.term()
        while self.peek() in {"+", "-"}:
            op = self.take()
            self.operations += 1
            node = Bin(str(op), node, self.term())
        return node

    def term(self) -> Node:
        node = self.unary()
        while self.peek() in {"*", "/", "mod", "%of"}:
            op = self.take()
            self.operations += 1
            node = Bin(str(op), node, self.unary())
        return node

    def unary(self) -> Node:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise NotArithmetic
        try:
            if self.peek() in {"-", "+"}:
                negate = self.take() == "-"
                inner = self.unary()
                return Neg(inner) if negate else inner
            return self.power()
        finally:
            self.depth -= 1

    def power(self) -> Node:
        base = self.atom()
        if self.peek() == "^":
            self.take()
            self.operations += 1
            return Bin("^", base, self.unary())
        return base

    def atom(self) -> Node:
        token = self.take()
        if isinstance(token, Decimal):
            return Num(token)
        if token == "(":
            self.depth += 1
            if self.depth > MAX_DEPTH:
                raise NotArithmetic
            node = self.expr()
            self.depth -= 1
            if self.take() != ")":
                raise NotArithmetic
            return node
        if token == "sqrt":
            self.operations += 1
            return Sqrt(self.unary())
        raise NotArithmetic


def _check(value: Decimal) -> Decimal:
    if not value.is_finite() or value.adjusted() >= MAX_DIGITS:
        raise NotArithmetic
    return value


def _run(node: Node) -> Decimal:
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Neg):
        return -_run(node.operand)
    if isinstance(node, Sqrt):
        value = _run(node.operand)
        if value < 0:
            raise NotArithmetic
        return _check(value.sqrt())
    left, right = _run(node.left), _run(node.right)
    match node.op:
        case "+":
            result = left + right
        case "-":
            result = left - right
        case "*":
            result = left * right
        case "/":
            result = left / right
        case "mod":
            result = left % right
        case "%of":
            result = left * right / 100
        case _:
            if right != right.to_integral_value() or abs(right) > MAX_EXPONENT:
                raise NotArithmetic
            result = left ** int(right)
    return _check(result)


def _format(value: Decimal) -> str:
    rounded = round(value, PLACES)
    if rounded == 0 and value != 0:
        rounded = _significant(value)  # a tiny result keeps its first digits instead of rounding to zero
    return "0" if rounded == 0 else format(rounded.normalize(), "f")


def _significant(value: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = PLACES
        return +value


def evaluate(text: str) -> str | None:
    """The answer to `text` as a plain decimal string, or None when `text` is not clearly arithmetic."""
    if not isinstance(text, str) or not 0 < len(text) <= MAX_CHARS:
        return None
    try:
        tree = _Parser(_tokens(text.strip().casefold().rstrip("?!. "))).parse()
        with localcontext() as context:
            context.prec = 2 * MAX_DIGITS
            return _format(_run(tree))
    except (NotArithmetic, ArithmeticError, RecursionError):
        return None
