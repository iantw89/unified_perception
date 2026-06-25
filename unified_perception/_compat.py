"""Small compatibility shims so the package degrades gracefully."""

from __future__ import annotations


def get_tqdm():
    """Return tqdm if available, else a no-op fallback with the same call shape."""
    try:
        from tqdm import tqdm

        return tqdm
    except ImportError:  # pragma: no cover
        class _NoTqdm:
            def __init__(self, iterable=None, **kwargs):
                self.iterable = iterable if iterable is not None else []

            def __iter__(self):
                return iter(self.iterable)

            def set_postfix(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return _NoTqdm
