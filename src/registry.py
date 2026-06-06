"""Component registry + factory — the swap-by-config mechanism.

This is what lets the pipeline swap any component (chunker, embedder, ...) by
changing ONE string in a config, with no code changes. Think of it as a wall
power socket: the socket shape is the interface; any appliance (lamp, charger)
with the right plug works; the config chooses which appliance is plugged in.

Three pieces:
  REGISTRY     a lookup table: REGISTRY[kind][name] -> the class
  @register    a decorator each implementation puts on itself to file into the table
  build        the factory: read a config's `type`/`params`, look up the class, make one

Pulled forward from the "wire everything" brick so the mechanism is real and
demonstrable from the very first component. Matches LLD §7.
"""

# kind ("chunker") -> name ("fixed") -> class (FixedChunker)
REGISTRY: dict[str, dict[str, type]] = {}


def register(kind: str, name: str):
    """Class decorator: file `cls` into REGISTRY[kind][name].

    Usage:
        @register("chunker", "fixed")
        class FixedChunker: ...
    The decorated class is returned unchanged, so it behaves normally otherwise.
    """
    def deco(cls):
        bucket = REGISTRY.setdefault(kind, {})
        if name in bucket and bucket[name] is not cls:
            raise ValueError(f"{kind!r} already has a component named {name!r}")
        bucket[name] = cls
        return cls
    return deco


def build(kind: str, type_name: str, params: dict | None = None):
    """Factory: instantiate the registered class for (kind, type_name) with params.

    e.g. build("chunker", "fixed", {"size": 512, "overlap": 50}) -> FixedChunker(...)
    Raises a clear error listing available names if the type is unknown.
    """
    bucket = REGISTRY.get(kind, {})
    if type_name not in bucket:
        available = sorted(bucket) or ["(none registered)"]
        raise KeyError(f"No {kind} named {type_name!r}. Available: {available}")
    return bucket[type_name](**(params or {}))


def available(kind: str) -> list[str]:
    """List the registered names for a kind (handy for tests / debugging)."""
    return sorted(REGISTRY.get(kind, {}))
