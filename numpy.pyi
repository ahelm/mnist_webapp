from typing import Any, Tuple

class ndarray:
    def reshape(self, shape: Tuple) -> "ndarray": ...

class dtype:
    def __init__(self, type: Any) -> None: ...

uint8 = dtype(int)

def frombuffer(
    buffer: bytes,
    dtype: dtype = dtype(float),
    count: int = -1,
    offset: int = 0,
) -> ndarray: ...
