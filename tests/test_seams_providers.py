"""Integration-seam audit, round 2: what a provider says (headers, bodies) is data, never a policy."""

from __future__ import annotations

from test_chain import Clock, Fake, chain, err, run

from glide.providers.chain import ChainPolicy
from glide.providers.config import GlideConfig
from glide.providers.errors import retry_after


def test_a_huge_retry_after_cannot_bench_a_slot_for_the_life_of_the_process():
    # finding 6: `retry-after: 99999999999` from one 429 rested the slot effectively forever
    assert retry_after({"retry-after": "99999999999"}) == 99999999999.0  # the header reader reports what it was told
    clock = Clock()
    c = chain(Fake(err("rate_limit", retry_after=1e9)), Fake("B"), clock=clock)
    assert run(c) == "B"
    rows = {r["name"]: r for r in c.status()}
    assert rows["a"]["resting_s"] == ChainPolicy().max_rest_s
    clock.advance(ChainPolicy().max_rest_s + 1)
    assert {r["name"]: r for r in c.status()}["a"]["resting_s"] == 0.0


def test_the_cap_is_a_policy_key_and_a_shorter_retry_after_is_kept():
    clock = Clock()
    c = chain(Fake(err("rate_limit", retry_after=500)), Fake("B"), policy=ChainPolicy(max_rest_s=60), clock=clock)
    run(c)
    assert {r["name"]: r for r in c.status()}["a"]["resting_s"] == 60.0
    c = chain(Fake(err("rate_limit", retry_after=5)), Fake("B"), policy=ChainPolicy(max_rest_s=60), clock=clock)
    run(c)
    assert {r["name"]: r for r in c.status()}["a"]["resting_s"] == 5.0
    config = GlideConfig.from_toml("[llm.fast]\nmax_rest_s = 120\nchain = ['openai:m']\n", env={})
    assert config.roles["llm.fast"].policy.max_rest_s == 120.0
