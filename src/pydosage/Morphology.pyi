import numpy as np
import numpy.typing as npt

class Morphology:

    _width: int
    _height: int
    marker: npt.NDArray[np.float32]
    mask: npt.NDArray[np.float32]
    work: npt.NDArray[np.float32]

    def __init__(
            self,
            width: int,
            height: int
    ) -> None: ...

    def get_width(self) -> int: ...

    def get_height(self) -> int: ...

    def reconstruct(self, method: str, niter: int) -> npt.NDArray[np.float32]: ...

Dilate: int
Erode: int