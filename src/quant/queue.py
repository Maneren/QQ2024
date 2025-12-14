from typing import cast
import numpy as np
from itertools import cycle


class RollingQueue:
    """Serve as custom version of queue."""

    def __init__(self, n: int) -> None:
        """Initialize queue."""
        self.size: int = n
        self.values = np.zeros((n,), dtype=np.float64)
        self.index = cycle(range(n))

    def put(self, value: np.float64) -> None:
        """Put new value in queue."""
        self.values[next(self.index) % len(self.values)] = value
        self.size = min(self.size + 1, len(self.values))

    def average(self) -> np.float64:
        """Return average value."""
        return (
            cast(np.float64, np.mean(self.values))
            if len(self.values) > 0
            else np.float64(0)
        )
