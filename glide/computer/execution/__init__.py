"""The structured execution engine: planned effects, observed targets, verified results.

`engine.run_execution` is the entry point (`runner.run` calls it for `engine="structured"`). Its backends (the
browser and desktop adapters) and the research supervisor are reached only through `Backend` (contracts.py) and
the two factories at the top of engine.py.
"""
