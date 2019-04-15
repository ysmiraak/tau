from collections.abc import Mapping
from functools import partial


def identity(x):
    """a -> a"""
    return x


def comp(g, f, *fs):
    """(b -> c) -> (a -> b) -> (a -> c)"""
    if fs: f = comp(f, *fs)
    return lambda x: g(f(x))


def diter(depth, xs):
    """like `iter` but yields items at `depth`."""
    if depth:
        for x in xs:
            yield from diter(x, depth= depth - 1)
    else:
        yield from xs


class Record(Mapping):
    """a `dict`-like type whose instances are partial finite mappings from
    attribute keys to arbitrary values.

    like a dict, a record is transparent about its content.

    unlike a dict, a record can access its content as object
    attributes without the hassle of string quoting.

    """

    def __init__(self, *records, **entries):
        for rec in records:
            for key, val in rec.items():
                setattr(self, key, val)
        for key, val in entries.items():
            setattr(self, key, val)

    def __repr__(self):
        return repr(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)


def select(record, *keys):
    """returns a Record with only entries under `keys` in `record`."""
    return Record({k: record[k] for k in keys})
