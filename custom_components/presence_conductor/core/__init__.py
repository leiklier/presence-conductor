"""Pure domain core — no ``homeassistant`` imports; see docs/ENGINE_SPEC.md.

The spec is the normative contract for the estimation core: the adapter
layer feeds the core sensor frames and monotonic ticks, the core returns
state changes and timer requests through a plan object. No wall clock, no
I/O. This boundary is enforced by ``tests/test_purity.py`` and by ruff's
TID251 ban scoped to this package.
"""
