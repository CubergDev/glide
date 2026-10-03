"""The chat adapter for any server that speaks OpenAI's Chat Completions API, and the facade over a chain of them.

OpenRouter, OpenAI, DeepSeek, Gemini's compatibility endpoint and local servers all take the same request and
differ only in what they refuse: a reasoning model wants `max_completion_tokens` and no temperature, one server
has no `json_schema` mode, another no `reasoning_effort`. So the client asks for the most it can, learns from each
400 what this endpoint accepts, remembers that for the life of the instance, and asks again. The schema is also
spelled out in the system prompt (the pattern of `computer/writer.py`, `_structured`) because a server may take
`response_format` and ignore it, and the reply is checked for JSON before it is returned, so a model that talked
instead of answering fails over (kind "content") rather than reaching the caller.

Everything that can go wrong leaves as a `ProviderError` (errors.py), never as an httpx error. Messages carry the
status and a short snippet of what the server said, with the key and anything that looks like an image scrubbed out
of it: some servers echo a rejected key or the start of a rejected payload. The key lives in a wrapper whose repr is
"<secret>", is sent per request and is never part of a URL, so `Slot`'s dataclass repr, a traceback or a log of the
client cannot print it.

The reply's reasoning is never returned: only `message.content` and `delta.content` are read, and a `<think>`
block that a server leaves inline in the content is cut out, also when the tag arrives split across stream chunks.
"""

from __future__ import annotations

import codecs
import contextlib
import copy
import json
import math
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from functools import partial
from urllib.parse import urlsplit

import httpx

from . import interrupt
from .base import ChatResult, LLMClient, ProviderSpec, Usage
from .chain import Chain
from .errors import ProviderError, from_exception, from_status, snippet, spent

DEFAULT_TIMEOUT_S = 30.0
CONNECT_TIMEOUT_S = 5.0
# httpx closes a connection that has been idle for 5 s, which is shorter than a spoken turn, so every first call of a
# turn would pay a new handshake. Kept longer, a connection can have been closed by the server meanwhile, which shows
# as the server hanging up before it answers; `_open` asks once more when it does.
KEEPALIVE_S = 30.0

# Most specific first. An endpoint that refuses one is asked with the next, for the rest of this client's life.
RESPONSE_FORMATS = ("json_schema", "json_object", "none")
FORMAT_WORDS = ("response_format", "json_schema", "json_object")
TOKEN_PARAMS = ("max_tokens", "max_completion_tokens")  # OpenAI's newer models only take the second
ADAPTABLE_STATUS = (400, 422)  # 422 is how FastAPI-based local servers say 400
# A first try, one retry for each thing a server can refuse (the token parameter, reasoning_effort, temperature and two
# response formats) and one for a connection that went stale. Every retry changes what is sent or is used up, so the
# loop ends; this is the belt to that brace.
MAX_ATTEMPTS = 7
STALE = (httpx.RemoteProtocolError, httpx.ReadError)
ERROR_BODY_LIMIT = 64 * 1024
SCHEMA_PROMPT = "Answer with a single JSON object and nothing else, matching this schema:"
OPEN_TAG, CLOSE_TAG = "<think>", "</think>"

_clock = time.monotonic  # a seam for tests
_LONG_BLOB = re.compile(r"[A-Za-z0-9+/=_-]{80,}")  # base64 of an image, or any other long token a server may echo
_BEARER = re.compile(r"(?i)bearer\s+\S+")
MIN_ECHO = 12  # shorter request text than this is too common a string to cut out of an error reply
_EOL = re.compile(r"\r\n|\n|\r")  # the line ends of the SSE spec, and no others
_HTTPX_ERRORS = (httpx.HTTPError, httpx.InvalidURL, httpx.StreamError)


class _Secret:
    """A string that never prints itself, so it cannot leak through a repr, a traceback or a log line."""

    __slots__ = ("_value",)

    def __init__(self, value: str | None):
        self._value = value or ""

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "<secret>"

    __str__ = __repr__


@dataclass(frozen=True)
class _Mode:
    """What this endpoint has shown it accepts. Replaced as a whole, under the client's lock, when it refuses something."""

    response_format: str = "json_schema"
    token_param: str = "max_tokens"
    token_param_switched: bool = False  # the switch happens once: a message that names both parameters cannot make it loop
    effort: bool = True  # False once the endpoint has refused reasoning_effort
    temperature: bool = True  # False once it has refused temperature (reasoning models take only their default)


@dataclass
class _Reply:
    """One parsed chat completion. `text` has its inline <think> block cut out."""

    text: str
    usage: Usage
    finish_reason: str | None
    logprobs: list[dict] | None
    refusal: str | None


class OpenAICompatLLM:
    """One OpenAI-compatible endpoint and model, as one slot of a chain. Implements `base.LLMClient`.

    One `httpx.Client` serves every call, so the connection (and its TLS session) is reused: the first
    call pays the handshake and the rest do not. Calls may overlap on several threads, as when a chain hedges.

    `options`, all optional, as written in glide.toml under the provider or the slot:
      extra_body       dict merged into every request body (a provider-routing block, a thinking switch). The
                       fields this client owns (model, messages, stream, the token limit) win over it.
      extra_headers    dict of headers sent with every request, after Authorization, so they may replace it.
      reasoning_effort sent as `reasoning_effort` when set: the way to turn thinking down. Dropped for good by an
                       endpoint that refuses it, and `settings` says so.
      token_param      "max_tokens" (default) or "max_completion_tokens". A 400 that names the other one flips it.
    Keys it does not know are ignored, because the same dict may carry settings meant for other layers.

    After a stream, `last_ttft_s`, `last_usage` and `last_finish_reason` describe the most recent one. They are
    per client, not per call, so two streams at once on the same client overwrite each other.
    """

    def __init__(
        self,
        name: str,
        model: str,
        base_url: str,
        api_key: str | None,
        options: Mapping | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        opts = dict(options or {})
        base = (base_url or "").strip().rstrip("/")
        parts = urlsplit(base)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"{name}: base_url must be an http(s) URL")
        token_param = opts.get("token_param") or TOKEN_PARAMS[0]
        if token_param not in TOKEN_PARAMS:
            raise ValueError(f"{name}: token_param must be one of {', '.join(TOKEN_PARAMS)}, not {token_param!r}")
        extra_body, extra_headers = opts.get("extra_body") or {}, opts.get("extra_headers") or {}
        if not isinstance(extra_body, dict) or not isinstance(extra_headers, dict):
            raise ValueError(f"{name}: extra_body and extra_headers must be tables")
        key = (api_key or "").strip()
        if key and not (key.isascii() and key.isprintable()):
            raise ValueError(f"{name}: the API key has characters that cannot be sent in a header")  # never says which

        self.name = name
        self.model = model
        self.base_url = base
        self.last_ttft_s: float | None = None
        self.last_usage: Usage | None = None
        self.last_finish_reason: str | None = None
        self._display = (
            f"{parts.scheme}://{parts.hostname}{f':{parts.port}' if parts.port else ''}{parts.path}"  # no userinfo, no query
        )
        self._url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        self._key = _Secret(key)
        self._effort: str | None = str(opts["reasoning_effort"]) if opts.get("reasoning_effort") else None
        self._extra_body = copy.deepcopy(extra_body)
        self._extra_headers = {str(k): str(v) for k, v in extra_headers.items()}
        self._mode = _Mode(token_param=token_param)
        self._lock = threading.Lock()
        self._client = httpx.Client(
            transport=transport,
            timeout=_timeout(DEFAULT_TIMEOUT_S),
            limits=httpx.Limits(keepalive_expiry=KEEPALIVE_S),
            follow_redirects=False,  # a redirect would carry the key to wherever it points
        )

    def __repr__(self) -> str:
        return f"OpenAICompatLLM(name={self.name!r}, base_url={self._display!r})"

    def __enter__(self) -> OpenAICompatLLM:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @property
    def settings(self) -> dict:
        """What this endpoint has been found to accept so far, for `glide doctor`."""
        with self._lock:
            mode = self._mode
        return {
            "response_format": mode.response_format,
            "token_param": mode.token_param,
            "reasoning_effort": self._effort if mode.effort else None,
            "temperature": mode.temperature,
        }

    # -- the LLMClient methods --------------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        schema: dict | None = None,
        logprobs: bool = False,
        timeout: float | None = None,
    ) -> ChatResult:
        """The whole reply as text. `timeout` bounds the wait for the answer, in total, and defaults to 30 s.

        With a `schema` the text is the JSON in the reply (a fence or a sentence around it is dropped), and a reply
        with none, or one without the schema's required fields, is a "content" error so another provider is tried.
        """
        limit = DEFAULT_TIMEOUT_S if timeout is None else timeout
        started = _clock()
        prepared = _with_schema(messages, schema) if schema is not None else list(messages)
        response = self._open(
            lambda mode: self._body(
                prepared, mode, max_tokens=max_tokens, temperature=temperature, schema=schema, logprobs=logprobs, stream=False
            ),
            schema=schema,
            stream=False,
            limit=limit,
            started=started,
            echo=_texts(prepared),
        )
        try:
            with interrupt.closing(partial(interrupt.abort_response, response), self.name):
                raw = self._drain(response, deadline=started + limit)
        except _HTTPX_ERRORS as e:
            interrupt.check(self.name)  # a connection closed by a cancel is not a transport fault
            raise from_exception(e, provider=self.name) from e
        finally:
            response.close()
        reply = self._completion(raw, want_logprobs=logprobs)
        if not reply.text:
            raise self._empty(reply.finish_reason, reply.refusal)
        text = self._json_in(reply.text, schema, reply.finish_reason) if schema is not None else reply.text
        return ChatResult(
            text=text,
            usage=reply.usage,
            provider=self.name,
            model=self.model,
            latency_s=_clock() - started,
            finish_reason=reply.finish_reason,
            logprobs=reply.logprobs,
        )

    def stream(
        self, messages: Sequence[dict], *, max_tokens: int = 512, temperature: float = 0.0, timeout: float | None = None
    ) -> Iterator[str]:
        """Text deltas as they arrive, never empty and never reasoning. A generator: the request goes out on the first
        `next()`, which is what a chain does at once, so an error before the first byte is raised there, as a
        ProviderError, and the chain moves on. Closing the generator early closes the connection.

        `timeout` bounds the wait for the first token (keep-alive comments do not extend it) and each later silence.
        """
        limit = DEFAULT_TIMEOUT_S if timeout is None else timeout
        started = _clock()
        self.last_ttft_s = self.last_usage = self.last_finish_reason = None
        prepared = list(messages)
        response = self._open(
            lambda mode: self._body(
                prepared, mode, max_tokens=max_tokens, temperature=temperature, schema=None, logprobs=False, stream=True
            ),
            schema=None,
            stream=True,
            limit=limit,
            started=started,
            echo=_texts(prepared),
        )
        try:
            with interrupt.closing(partial(interrupt.abort_response, response), self.name):
                yield from self._events(response, started, started + limit)
        except _HTTPX_ERRORS as e:
            interrupt.check(self.name)
            raise from_exception(e, provider=self.name) from e
        finally:
            response.close()

    # -- the request ------------------------------------------------------------------------------

    def _headers(self, stream: bool) -> dict[str, str]:
        headers = {"Accept": "text/event-stream" if stream else "application/json"}
        if key := self._key.reveal():
            headers["Authorization"] = f"Bearer {key}"
        return {**headers, **self._extra_headers}

    def _body(
        self,
        messages: list[dict],
        mode: _Mode,
        *,
        max_tokens: int,
        temperature: float,
        schema: dict | None,
        logprobs: bool,
        stream: bool,
    ) -> dict:
        body = copy.deepcopy(self._extra_body)  # what is set below wins over it
        body.update(model=self.model, messages=messages, stream=stream)
        body[mode.token_param] = max_tokens
        if mode.temperature:
            body["temperature"] = temperature
        if mode.effort and self._effort:
            body["reasoning_effort"] = self._effort
        if (response_format := _response_format(mode.response_format, schema)) is not None:
            body["response_format"] = response_format
        if logprobs:
            body["logprobs"] = True
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def _open(
        self,
        build: Callable[[_Mode], dict],
        *,
        schema: dict | None,
        stream: bool,
        limit: float,
        started: float,
        echo: Sequence[str] = (),
    ):
        """Send the request and return the open 2xx response, which the caller must close.

        A 400 that names something this endpoint refuses changes what is sent (see `_adapt`) and the request is
        made again. Any other non-2xx answer is raised as the ProviderError its status maps to.
        """
        error: ProviderError | None = None
        stale_retried = False
        for _ in range(MAX_ATTEMPTS):
            interrupt.check(self.name)
            with self._lock:
                mode = self._mode
            call = interrupt.Call(self._client)
            request = self._client.build_request(
                "POST",
                self._url,
                json=build(mode),
                headers=self._headers(stream),
                timeout=_timeout(limit),
                extensions=call.extensions,
            )
            sent = _clock()
            try:
                with interrupt.closing(call.abort, self.name):  # a model's answer is awaited here, before any header
                    response = self._client.send(request, stream=True)
            except _HTTPX_ERRORS as e:
                interrupt.check(self.name)
                # A kept-alive connection that the server had closed fails at once, and a fresh one will do. A reset
                # after a long wait is the server giving up on a real request: asking again would break the timeout
                # and may be billed twice.
                quick = _clock() - sent <= CONNECT_TIMEOUT_S and _clock() < started + limit
                if isinstance(e, STALE) and not stale_retried and quick:
                    stale_retried = True
                    continue
                raise from_exception(e, provider=self.name) from e
            with interrupt.closing(partial(interrupt.abort_response, response), self.name):  # a cancel while the answer is read
                if response.is_success:
                    return response
                text = _without(self._error_text(response, started + limit), echo)
            error = self._status_error(response.status_code, text, response.headers)
            sent_format = schema is not None and mode.response_format != "none"
            if response.status_code in ADAPTABLE_STATUS and self._adapt(mode, text.lower(), sent_format):
                continue
            raise error
        raise error  # the loop ran at least once

    def _adapt(self, used: _Mode, said: str, sent_format: bool) -> bool:
        """After a 400 to a request made with `used`: True when asking again in a different way is worth it.

        Compare-and-set under the lock: two calls refused at once must not both step down, or the second would skip
        `json_object`. If another call already changed the mode, this one just asks again with the new mode.
        """
        with self._lock:
            if self._mode != used:
                return True
            other = TOKEN_PARAMS[1 - TOKEN_PARAMS.index(used.token_param)]
            if not used.token_param_switched and other in said:
                new = replace(used, token_param=other, token_param_switched=True)
            elif used.effort and self._effort and "reasoning_effort" in said:
                new = replace(used, effort=False)
            elif used.temperature and "temperature" in said:
                new = replace(used, temperature=False)
            elif sent_format and used.response_format != "none" and any(word in said for word in FORMAT_WORDS):
                new = replace(used, response_format=RESPONSE_FORMATS[RESPONSE_FORMATS.index(used.response_format) + 1])
            else:
                return False
            self._mode = new
            return True

    # -- errors -----------------------------------------------------------------------------------

    def _scrub(self, text: str) -> str:
        """`text` without this client's key and without anything that looks like an encoded image or token."""
        if key := self._key.reveal():
            text = text.replace(key, "[redacted]")
        return _LONG_BLOB.sub("[...]", text)

    def _status_error(self, status: int, body: str, headers: Mapping[str, str]) -> ProviderError:
        # A spent account (a 402, or a 429 that says so, in the code of the reply) comes back from errors.from_status
        # as `auth` with fixed text, so it is judged on the whole reply. Any other message is only what the server said.
        said = self._scrub(body) if spent(status, body) else _what_it_said(self._scrub(body))
        return from_status(status, said, provider=self.name, retry_after=_retry_after(headers))

    def _body_error(self, error: object) -> ProviderError:
        """An error a server reported inside a 200 reply or a stream (OpenRouter does), mapped as its status would be."""
        message = error.get("message") if isinstance(error, dict) else error
        code = error.get("code") if isinstance(error, dict) else None
        text = message if isinstance(message, str) else json.dumps(message)
        try:
            status = int(code)
        except (TypeError, ValueError):
            status = 0
        if 400 <= status < 600:
            return self._status_error(status, text, {})
        return ProviderError(
            f"{self.name} reported an error: {snippet(self._scrub(text))}".rstrip(": "), kind="server", provider=self.name
        )

    def _content_error(self, detail: str) -> ProviderError:
        return ProviderError(f"{self.name} {detail}", kind="content", provider=self.name)

    def _empty(self, finish_reason: str | None, refusal: str | None) -> ProviderError:
        """A reply with nothing in it to use."""
        if finish_reason == "length":
            return self._content_error(
                "answered with no text and stopped at the token limit: the budget was likely spent on thinking "
                "(raise max_tokens or turn reasoning_effort down)"
            )
        if refusal:
            return self._content_error(f"refused to answer: {snippet(self._scrub(refusal))}")
        return self._content_error(f"answered with no text (finish_reason {finish_reason})")

    def _error_text(self, response: httpx.Response, deadline: float) -> str:
        """The body of a failed response, closing it. What could not be read is simply missing: the status is the news."""
        try:
            return self._scrub(self._drain(response, limit=ERROR_BODY_LIMIT, deadline=deadline).decode("utf-8", "replace"))
        except (*_HTTPX_ERRORS, ProviderError):
            return ""
        finally:
            response.close()

    def _drain(self, response: httpx.Response, *, limit: int | None = None, deadline: float | None = None) -> bytes:
        """The body, read in pieces so the deadline holds against a server that dribbles bytes forever."""
        out = bytearray()
        for chunk in response.iter_bytes():
            out += chunk
            if limit is not None and len(out) >= limit:
                break
            if deadline is not None and _clock() > deadline:
                raise ProviderError(f"{self.name} timed out", kind="timeout", provider=self.name)
        return bytes(out)

    # -- reading a reply --------------------------------------------------------------------------

    def _json(self, raw: bytes) -> dict:
        try:
            data = json.loads(raw)
        except ValueError:
            raise self._content_error("answered with something that is not JSON") from None
        if not isinstance(data, dict):
            raise self._content_error("answered with JSON that is not a chat completion")
        return data

    def _completion(self, raw: bytes, *, want_logprobs: bool) -> _Reply:
        """A non-streaming reply's text, usage and finish reason; its reasoning fields are never looked at."""
        data = self._json(raw)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            if data.get("error"):  # a 200 with the error in the body
                raise self._body_error(data["error"])
            raise self._content_error("answered without choices")
        choice = choices[0]
        finish = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
        if choice.get("error") or finish == "error":
            raise self._body_error(choice.get("error") or "the model stopped with an error")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise self._content_error("answered without a message")
        logprobs = choice.get("logprobs")
        entries = logprobs.get("content") if want_logprobs and isinstance(logprobs, dict) else None
        refusal = message.get("refusal")
        return _Reply(
            text=_strip_think(_text(message.get("content"))),
            usage=_usage(data.get("usage")),
            finish_reason=finish,
            logprobs=[e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else None,
            refusal=refusal if isinstance(refusal, str) else None,
        )

    def _json_in(self, text: str, schema: dict, finish_reason: str | None) -> str:
        """The first JSON value in `text` that fits the schema's top level. Models that were asked for JSON usually
        return exactly that; some wrap it in a fence or a sentence. A value that does not fit is skipped whole, so
        an object nested inside it is never mistaken for the answer."""
        decoder = json.JSONDecoder()
        position = 0
        while (start := _next_json_start(text, position)) >= 0:
            try:
                value, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                position = start + 1
                continue
            if _fits(value, schema):
                return text[start:end]
            position = end
        cut = " (the reply was cut off at the token limit)" if finish_reason == "length" else ""
        raise self._content_error(f"answered without JSON that fits the schema{cut}")  # never the reply itself

    def _events(self, response: httpx.Response, started: float, deadline: float) -> Iterator[str]:
        """The text deltas of a stream, or of a plain JSON reply from a server that ignored `stream`."""
        if "json" in response.headers.get("content-type", "").lower():
            reply = self._completion(self._drain(response, deadline=deadline), want_logprobs=False)
            if not reply.text:
                raise self._empty(reply.finish_reason, reply.refusal)
            self.last_usage, self.last_finish_reason = reply.usage, reply.finish_reason
            self.last_ttft_s = _clock() - started
            yield reply.text
            return

        think = _ThinkFilter()
        emitted = done = False
        finish: str | None = None
        refusal: str | None = None

        body = response.iter_bytes()

        def lines() -> Iterator[str]:
            for line in _split_lines(body):
                if not emitted and _clock() > deadline:  # keep-alive comments must not hold the first token off forever
                    raise ProviderError(f"{self.name} timed out", kind="timeout", provider=self.name)
                yield line

        for event, data in _sse(lines()):
            if data.strip() == "[DONE]":
                done = True
                break
            if not data.strip():
                continue
            try:
                chunk = json.loads(data)
            except ValueError:
                raise self._content_error("sent a stream event that is not JSON") from None
            if not isinstance(chunk, dict):
                raise self._content_error("sent a stream event that is not an object")
            if event == "error" or chunk.get("error"):
                raise self._body_error(chunk.get("error") or chunk)
            if isinstance(chunk.get("usage"), dict):
                self.last_usage = _usage(chunk["usage"])
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                continue  # the usage-only chunk at the end has none
            choice = choices[0]
            if choice.get("error") or choice.get("finish_reason") == "error":
                raise self._body_error(choice.get("error") or "the model stopped with an error")
            if isinstance(choice.get("finish_reason"), str):
                finish = choice["finish_reason"]
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            if isinstance(delta.get("refusal"), str):
                refusal = delta["refusal"]
            # Only `content`: reasoning arrives as reasoning_content, reasoning or reasoning_details and is skipped.
            if text := think.feed(_text(delta.get("content"))):
                if not emitted:
                    emitted = True
                    self.last_ttft_s = _clock() - started
                yield text
        if text := think.flush():
            if not emitted:
                emitted = True
                self.last_ttft_s = _clock() - started
            yield text
        if done:
            # Read on to the end of the body, which is only its closing frame, so that the connection goes back to
            # the pool: closed half-read it would be dropped, and a stream per question would pay a handshake each.
            with contextlib.suppress(*_HTTPX_ERRORS):
                spent = 0
                for piece in body:
                    spent += len(piece)
                    if spent > ERROR_BODY_LIMIT:
                        break
        if not (done or finish):
            raise ProviderError(f"{self.name} ended the stream without finishing it", kind="transport", provider=self.name)
        self.last_finish_reason = finish
        if not emitted:
            raise self._empty(finish, refusal)


# -- helpers --------------------------------------------------------------------------------------


def _texts(messages: Sequence[dict]) -> list[str]:
    """Every piece of text in a request, longest first, so an error reply that quotes it can be cleaned."""
    found: list[str] = []
    for message in messages:
        content = message.get("content")
        parts = content if isinstance(content, list) else [content]
        for part in parts:
            text = part.get("text") if isinstance(part, dict) else part
            if isinstance(text, str) and len(text) >= MIN_ECHO:
                found.append(text)
    return sorted(set(found), key=len, reverse=True)


def _without(text: str, echo: Sequence[str]) -> str:
    """`text` with each piece of the request that it quotes cut out: a request body must not reach an error message.

    A server may quote the request inside a JSON string, where quotes and newlines are escaped, so that form is cut too.
    """
    for piece in echo:
        for form in (piece, json.dumps(piece)[1:-1]):
            text = text.replace(form, "[request text]")
    return text


def _what_it_said(body: str) -> str:
    """The message of an error reply. A JSON reply is reduced to its `error.message` (or `message`): the rest of it
    may be a quoted request, headers or credentials, which no error message may carry. Anything else is kept as is
    (a plain-text reply), after a bearer token, if one was quoted, is cut out."""
    try:
        data = json.loads(body)
    except ValueError:
        return _BEARER.sub("Bearer [redacted]", body)
    error = data.get("error", data) if isinstance(data, dict) else data
    message = error.get("message") if isinstance(error, dict) else error
    return _BEARER.sub("Bearer [redacted]", message) if isinstance(message, str) else "(no message in the reply)"


def _timeout(limit: float) -> httpx.Timeout:
    """Connect in 5 s at most; every read, the write and the pool wait take `limit`."""
    return httpx.Timeout(limit, connect=min(CONNECT_TIMEOUT_S, limit))


def _response_format(kind: str, schema: dict | None) -> dict | None:
    """The `response_format` to send, or None. With no schema there is nothing to constrain: plain chat stays plain."""
    if schema is None or kind == "none":
        return None
    if kind == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": schema}}
    return {"type": "json_object"}  # this mode needs the word JSON in the messages, which `_with_schema` adds


def _with_schema(messages: Sequence[dict], schema: dict) -> list[dict]:
    """A copy of the messages with the schema spelled out in the system prompt, whatever `response_format` does.

    The caller's messages are never changed: a chain passes the same list to each provider in turn, and a hedge
    to two at once.
    """
    note = f"{SCHEMA_PROMPT}\n{json.dumps(schema)}"
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        content = out[0].get("content")
        if isinstance(content, str):
            out[0]["content"] = f"{content}\n\n{note}" if content else note
        elif isinstance(content, list):
            out[0]["content"] = [*content, {"type": "text", "text": note}]
        else:
            out[0]["content"] = note
    else:
        out.insert(0, {"role": "system", "content": note})
    return out


def _next_json_start(text: str, position: int) -> int:
    starts = [i for i in (text.find("{", position), text.find("[", position)) if i >= 0]
    return min(starts) if starts else -1


def _fits(value: object, schema: dict) -> bool:
    """The top level of `value` against the schema: its type, and the required fields of an object. A full JSON
    Schema check is the server's job when it enforces the schema; this catches a reply that is plainly something else."""
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        return isinstance(value, dict) and all(name in value for name in schema.get("required", ()))
    if kind == "array":
        return isinstance(value, list)
    return True


def _text(content: object) -> str:
    """A message's content as text: a string, or a list of parts of which the text ones count."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text", "output_text") and isinstance(part.get("text"), str)
        )
    return ""


def _int(value: object) -> int:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) and value > 0 else 0


def _usage(raw: object) -> Usage:
    """Chat Completions usage with the cached tokens apart: `prompt_tokens` counts them too (as openai_writer._usage
    does). DeepSeek reports the cached part as `prompt_cache_hit_tokens` instead of in `prompt_tokens_details`."""
    if not isinstance(raw, dict):
        return Usage()
    prompt = _int(raw.get("prompt_tokens", raw.get("input_tokens")))
    details = raw.get("prompt_tokens_details")
    cached = (_int(details.get("cached_tokens")) if isinstance(details, dict) else 0) or _int(raw.get("prompt_cache_hit_tokens"))
    return Usage(
        input_tokens=max(prompt - cached, 0),
        output_tokens=_int(raw.get("completion_tokens", raw.get("output_tokens"))),
        cached_input_tokens=cached,
    )


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds the server asked us to wait: `retry-after-ms` (OpenAI sends it), or `retry-after` as seconds or a date."""
    get = headers.get
    try:
        if ms := get("retry-after-ms"):
            seconds = float(ms) / 1000
        elif value := get("retry-after"):
            try:
                seconds = float(value)
            except ValueError:
                when = parsedate_to_datetime(value)
                seconds = (when if when.tzinfo else when.replace(tzinfo=UTC)).timestamp() - datetime.now(UTC).timestamp()
        else:
            return None
    except (TypeError, ValueError):
        return None
    return max(seconds, 0.0) if math.isfinite(seconds) else None


def _split_lines(chunks: Iterable[bytes]) -> Iterator[str]:
    """Lines of an SSE body, split on \\n, \\r\\n and \\r only. httpx's `iter_lines` follows `str.splitlines`, which
    also splits on U+2028 and U+0085, and a model's text may hold them inside a JSON string."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending, first = "", True
    for chunk in chunks:
        text = pending + decoder.decode(chunk)
        if first and text:
            text, first = text.removeprefix("\ufeff"), False  # the spec drops a byte order mark at the very start
        held = text.endswith("\r")  # the \n of a \r\n may be in the next chunk
        parts = _EOL.split(text[:-1] if held else text)
        pending = parts.pop() + ("\r" if held else "")
        yield from parts
    pending += decoder.decode(b"", final=True)
    if pending.rstrip("\r"):
        yield pending.rstrip("\r")


def _sse(lines: Iterator[str]) -> Iterator[tuple[str, str]]:
    """(event, data) of each server-sent event. Comment lines (": keepalive") and other fields are skipped, `data:`
    may or may not have a space after the colon, and an event that the stream ends in the middle of is still delivered."""
    event, data = "", []
    for line in lines:
        if line == "":
            if data:
                yield event, "\n".join(data)
            event, data = "", []
        elif not line.startswith(":"):
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "data":
                data.append(value)
            elif field == "event":
                event = value
    if data:
        yield event, "\n".join(data)


class _ThinkFilter:
    """Cuts `<think>...</think>` out of text that arrives in pieces, the tag possibly split between two of them.

    A tail that could still turn into a tag is held back until the next piece settles it, so "a <" then " b" gives
    "a < b" and "<thi" then "nk>x</think>y" gives "y". A block still open at the end is dropped (the model ran out
    of tokens thinking); a held-back "<" outside a block was text and is released by `flush`. The whitespace a model
    puts after `</think>` (usually "\n\n") is dropped when the answer would otherwise start with it, or the text
    before the block already ended in some; "One<think>x</think> two" keeps its space.
    """

    def __init__(self) -> None:
        self._inside = False
        self._after_space = True  # nothing has been shown, or what was shown ended in whitespace
        self._trim = False  # a block has just closed where that is so: drop the whitespace that follows
        self._held = ""

    def feed(self, text: str) -> str:
        buffer, self._held = self._held + text, ""
        out: list[str] = []
        while buffer:
            tag = CLOSE_TAG if self._inside else OPEN_TAG
            found = re.search(re.escape(tag), buffer, re.IGNORECASE)
            if found:
                if self._inside:
                    self._trim = self._after_space
                else:
                    self._visible(out, buffer[: found.start()])
                buffer = buffer[found.end() :]
                self._inside = not self._inside
                continue
            keep = _partial_tag(buffer, tag)
            if not self._inside:
                self._visible(out, buffer[: len(buffer) - keep])
            self._held = buffer[len(buffer) - keep :]
            break
        return "".join(out)

    def _visible(self, out: list[str], piece: str) -> None:
        """Text outside a block, minus the whitespace that follows a block that has just closed."""
        if self._trim:
            piece = piece.lstrip()
            self._trim = not piece
        if piece:
            self._after_space = piece[-1].isspace()
        out.append(piece)

    def flush(self) -> str:
        held, self._held = ("" if self._inside else self._held), ""
        return held


def _partial_tag(buffer: str, tag: str) -> int:
    """How many characters at the end of `buffer` are the start of `tag`."""
    for size in range(min(len(tag) - 1, len(buffer)), 0, -1):
        if tag.startswith(buffer[-size:].lower()):
            return size
    return 0


def _strip_think(text: str) -> str:
    think = _ThinkFilter()
    return (think.feed(text) + think.flush()).strip()


# -- the facade -----------------------------------------------------------------------------------


class LLM:
    """The LLM as the rest of Glide sees it: a chain of clients behind the same two methods, so no caller knows
    which vendor answered. A failure moves on to the next client and is recorded on `chain.events`."""

    def __init__(self, chain: Chain[LLMClient]):
        self.chain = chain

    def chat(self, messages: Sequence[dict], *, hedge: bool | None = None, **kw) -> ChatResult:
        """`kw` are those of `LLMClient.chat`. `ChatResult.provider` is the name of the slot that answered.

        `hedge=None` races the next slot against a slow one when the chain's policy sets `hedge_after_s`, and
        does nothing otherwise; True asks for that explicitly and False never races. A chat is safe to run twice.
        """
        race = self.chain.policy.hedge_after_s is not None if hedge is None else hedge
        return self.chain.call(lambda slot: replace(slot.client.chat(messages, **kw), provider=slot.name), hedge=race)

    def stream(self, messages: Sequence[dict], **kw) -> Iterator[str]:
        """Text deltas, from the first slot that produces one. Once a delta has been yielded a failure is a
        ProviderError of kind "stream" and nothing else is tried: the caller has already used what came before."""
        return self.chain.stream(lambda slot: slot.client.stream(messages, **kw))


def build_client(spec: ProviderSpec, model: str, api_key: str | None, options: dict) -> OpenAICompatLLM:
    """The client for one slot, named "<provider>:<model>". The provider's own options sit under the slot's."""
    return OpenAICompatLLM(f"{spec.name}:{model}", model, spec.base_url, api_key, {**spec.options, **(options or {})})
