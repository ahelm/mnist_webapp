from pathlib import Path
from typing import Optional

import numpy as np

class Image:
    def save(self, file: Path) -> None: ...

def from_array(arr: np.ndarray, mode: Optional[str] = None) -> Image: ...
