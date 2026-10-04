"""The structured execution engine: planned effects, observed targets, verified results.

`engine.run_execution` is the entry point (`runner.run` calls it for `engine="structured"`). Its backends (the
browser and desktop adapters) are built only by `providers.make_backend` and reached through `Backend` (contracts.py);
the research supervisor is `research.Supervisor`.
"""
