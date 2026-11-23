"""Identity-track helpers shared by data providers."""

from __future__ import annotations

import numpy as np


def fill_identity_gaps(observed: np.ndarray, max_gap: int) -> np.ndarray:
    """Existence per (frame, person): observed, or inside a short unobserved gap.

    ``observed`` is (F, P) bool over a whole recording (never a window: whether
    a person comes back is only known from the full sequence).  A run of at most
    ``max_gap`` unobserved frames *between two observations of the same person*
    is marked present.  Leading/trailing absences and longer gaps are not.
    """
    observed = np.asarray(observed, dtype=bool)
    present = observed.copy()
    if max_gap <= 0:
        return present
    for person in range(observed.shape[1]):
        frames = np.flatnonzero(observed[:, person])
        if len(frames) < 2:
            continue
        gaps = np.diff(frames) - 1
        for first, gap in zip(frames[:-1], gaps):
            if 0 < gap <= max_gap:
                present[first + 1:first + 1 + gap, person] = True
    return present
