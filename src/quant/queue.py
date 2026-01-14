from typing import cast
import numpy as np


class RollingQueue:
    """Serve as custom version of queue."""

    def __init__(self, n: int) -> None:
        """Initialize queue."""
        self.values = np.zeros((n,), dtype=np.float64)
        self.index = 0

    def put(self, value: np.float64) -> None:
        """Put new value in queue."""
        self.values[self.index % len(self.values)] = value
        self.index += 1

    def average(self) -> np.float64:
        """Return average value."""
        return (
            cast(np.float64, np.mean(self.values)) if self.index > 0 else np.float64(0)
        )
