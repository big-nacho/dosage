import cv2
import numpy as np
import numpy.typing as npt

Dilate = 'dilate'
Erode = 'erode'
Kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def get_processor(method):
    return {
        Dilate: cv2.dilate,
        Erode: cv2.erode
    }[method]


def get_constraint(method):
    return {
        Dilate: np.minimum,
        Erode: np.maximum
    }[method]

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
    ) -> None:
        self._width = width
        self._height = height

        shape = (height, width)
        self.marker = np.zeros(shape, dtype=np.float32)
        self.mask = np.zeros(shape, dtype=np.float32)
        self.work = np.zeros(shape, dtype=np.float32)

    def get_width(self) -> int:
        return self._width

    def get_height(self) -> int:
        return self._height

    def reconstruct(self, method: str, niter: int) -> npt.NDArray[np.float32]:
        processor = get_processor(method)
        constraint = get_constraint(method)

        marker = self.marker
        mask = self.mask
        work = self.work
        for _ in range(niter):
            processor(marker, Kernel, work)
            constraint(work, mask, out=marker)

        return marker
