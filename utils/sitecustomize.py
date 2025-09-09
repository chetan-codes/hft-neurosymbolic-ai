# Compatibility shim for Python 3.10+ packages that import from `collections`
# rather than `collections.abc` (e.g., unification, kanren dependencies).
try:
    import collections
    import collections.abc as _abc
    for _name in ("Iterator", "Mapping", "MutableMapping", "Sequence", "Set", "Hashable"):
        if not hasattr(collections, _name) and hasattr(_abc, _name):
            setattr(collections, _name, getattr(_abc, _name))
except Exception:
    # Fail open; if anything goes wrong, don't block app startup
    pass
