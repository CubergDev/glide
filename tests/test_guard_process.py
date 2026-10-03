"""Processes: nothing starts, replaces itself, or signals another process from a test.

Each check first asserts the target is the guard's refusal, so a broken guard fails the assertion instead of forking,
replacing the test process, or opening a browser.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import subprocess
import sys
import webbrowser

import pytest
from conftest import _PROCESS_CALLS, _Refused

PRESENT = [name for name in _PROCESS_CALLS if hasattr(os, name)]


@pytest.mark.parametrize("name", PRESENT)
def test_os_process_calls_refuse(name):
    target = getattr(os, name)
    assert isinstance(target, _Refused), f"os.{name} is live in a test"
    with pytest.raises(RuntimeError, match="real machine"):
        target("/nonexistent", ["x"])


def test_the_process_calls_the_guard_names_cover_fork_exec_spawn_and_popen():
    for expected in ("fork", "execv", "spawnv", "popen", "killpg"):
        assert expected in _PROCESS_CALLS


@pytest.mark.parametrize("name", ["system", "posix_spawn", "posix_spawnp"])
def test_the_original_process_refusals_are_unchanged(name):
    if hasattr(os, name):
        with pytest.raises(RuntimeError, match="real machine"):
            getattr(os, name)("true")


def test_popen_and_everything_built_on_it_refuses():
    with pytest.raises(RuntimeError, match="real machine"):
        subprocess.Popen([sys.executable, "-c", "pass"])
    for call in (subprocess.run, subprocess.call, subprocess.check_call, subprocess.check_output):
        with pytest.raises(RuntimeError, match="real machine"):
            call([sys.executable, "-c", "pass"])
    with pytest.raises(RuntimeError, match="real machine"):
        os.popen("true")


def test_asyncio_subprocesses_refuse():
    with pytest.raises(RuntimeError, match="real machine"):
        subprocess.Popen(["true"])  # the asyncio transport ends here; a live Popen would fail this line first

    async def spawn():
        await asyncio.create_subprocess_exec(sys.executable, "-c", "pass")

    with pytest.raises(RuntimeError, match="real machine"):
        asyncio.run(spawn())

    async def spawn_shell():
        await asyncio.create_subprocess_shell("true")

    with pytest.raises(RuntimeError, match="real machine"):
        asyncio.run(spawn_shell())


def test_multiprocessing_refuses_to_start_a_process():
    assert isinstance(multiprocessing.process.BaseProcess.start, _Refused)
    process = multiprocessing.Process(target=print)
    with pytest.raises(RuntimeError, match="real machine"):
        process.start()


def test_another_process_cannot_be_signalled_but_this_one_and_pid_probes_can():
    with pytest.raises(RuntimeError, match="another process"):
        os.kill(1, signal.SIGTERM)
    with pytest.raises(RuntimeError, match="another process"):
        os.kill(os.getppid(), signal.SIGINT)
    os.kill(os.getpid(), 0)  # a test may probe itself
    handled = []
    previous = signal.signal(signal.SIGUSR1, lambda *args: handled.append(args))
    try:
        os.kill(os.getpid(), signal.SIGUSR1)  # and signal itself
    finally:
        signal.signal(signal.SIGUSR1, previous)
    assert handled


@pytest.mark.parametrize("name", ["open", "open_new", "open_new_tab", "get"])
def test_the_browser_is_never_opened(name):
    target = getattr(webbrowser, name)
    assert isinstance(target, _Refused), f"webbrowser.{name} is live in a test"
    with pytest.raises(RuntimeError, match="real machine"):
        target("http://127.0.0.1/")
