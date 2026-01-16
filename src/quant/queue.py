from collections import deque
import numpy as np
from numpy.typing import NDArray


class RollingQueue:
    """Serve as custom version of queue."""

    def __init__(self, n: int) -> None:
        """Initialize queue."""
        self.__values: deque[np.float64] = deque(maxlen=n)
        self.__result = np.zeros((n,), dtype=np.float64)
        self.__n = n
        self.__size = 0

    def put(self, value: np.float64) -> None:
        """Put new value in queue."""
        self.__values.append(value)
        self.__size = min(self.__size + 1, self.__n)

    @property
    def values(self) -> NDArray[np.float64]:
        if self.__size:
            self.__result[-self.__size :] = self.__values

        return self.__result
