"""Run a lazy callback workflow in the caller's synchronous or asynchronous context."""

import asyncio
import inspect


def _sync(value):
    if inspect.isawaitable(value):
        if inspect.iscoroutine(value):
            value.close()
        elif isinstance(value, asyncio.Future):
            value.cancel()
        raise TypeError("async callback requires the asynchronous entry point")
    return value


def drive(steps):
    value, error = None, None
    try:
        while True:
            try:
                callback = steps.throw(error) if error is not None else steps.send(value)
            except StopIteration as done:
                return done.value
            try:
                value, error = _sync(callback()), None
            except BaseException as failure:
                error = failure
    finally:
        steps.close()


async def adrive(steps):
    value, error = None, None
    try:
        while True:
            try:
                callback = steps.throw(error) if error is not None else steps.send(value)
            except StopIteration as done:
                return done.value
            try:
                value = callback()
                if inspect.isawaitable(value):
                    value = await value
                error = None
            except BaseException as failure:
                error = failure
    finally:
        steps.close()
