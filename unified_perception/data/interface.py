"""Data provider interface.

A provider discovers sequence/frame samples and returns tensors for one temporal
window.  This isolates dataset access from model/training code and enables
clean switching between backends (e.g. preprocessed export vs direct
human_prior-aligned folders).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Tuple


SampleRef = Tuple[str, int]


class DataProvider(ABC):
    @abstractmethod
    def list_samples(self) -> List[SampleRef]:
        """Return list of (sequence_id, start_frame)."""

    @abstractmethod
    def load_window(self, sample: SampleRef) -> dict[str, Any]:
        """Load one temporal sample window and return model-ready arrays/tensors."""
