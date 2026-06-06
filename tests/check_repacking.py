"""OFFLINE checks for repacking (src/repacking/). Pure ordering, no model.

Run directly:  python tests/check_repacking.py
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src  # noqa: F401
from src.registry import build, available
from src.repacking import Repacker, ForwardRepacker, ReverseRepacker, SidesRepacker
from src.indexing import RetrievedChunk
from src.chunking import Chunk


def _ranked(n):
    """n chunks ranked best→worst: ids 0-0 .. 0-(n-1)."""
    return [RetrievedChunk(chunk=Chunk(text=f"c{i}", doc_id="0", chunk_id=f"0-{i}"),
                           score=1.0 - i * 0.1, rank=i) for i in range(n)]


def test_registered():
    assert set(available("repacker")) == {"forward", "reverse", "sides"}


def test_forward_is_identity():
    c = _ranked(5)
    out = ForwardRepacker().pack(c)
    assert [x.chunk.chunk_id for x in out] == ["0-0", "0-1", "0-2", "0-3", "0-4"]


def test_reverse_flips():
    out = ReverseRepacker().pack(_ranked(5))
    assert [x.chunk.chunk_id for x in out] == ["0-4", "0-3", "0-2", "0-1", "0-0"]


def test_sides_strong_at_both_ends():
    out = SidesRepacker().pack(_ranked(5))
    # ranks 0,2,4 to the front; 1,3 to the tail reversed → 0,2,4,3,1
    assert [x.chunk.chunk_id for x in out] == ["0-0", "0-2", "0-4", "0-3", "0-1"]


def test_all_preserve_membership_and_count():
    # Reordering must never add/drop chunks.
    c = _ranked(7)
    ids = {x.chunk.chunk_id for x in c}
    for t in ("forward", "reverse", "sides"):
        out = build("repacker", t).pack(c)
        assert len(out) == len(c)
        assert {x.chunk.chunk_id for x in out} == ids


def test_interface_and_edge_cases():
    for t in ("forward", "reverse", "sides"):
        r = build("repacker", t)
        assert isinstance(r, Repacker)
        assert r.pack([]) == []                 # empty in → empty out
        one = _ranked(1)
        assert [x.chunk.chunk_id for x in r.pack(one)] == ["0-0"]   # single chunk


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} repacking checks passed.")


if __name__ == "__main__":
    _run()
