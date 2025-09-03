import cv2
import numpy as np
from .Morphology import Morphology, Dilate, Erode
from typing import Literal
import numpy.typing as npt

from .dosage import (
    detect,
    method_dosage,
    method_fast_mbd,
    method_hybrid
)


class Dosage:

    Dosage: int = method_dosage
    Fast_MBD: int = method_fast_mbd
    Hybrid: int = method_hybrid

    _width: int
    _height: int
    _area: int
    _shape_hw: tuple[int, int]
    _shape_wh = tuple[int, int]
    _method: int
    _n_iter: int
    _sigma: float
    _boundary_size: int | Literal['auto']
    _reconstruct: bool
    _reconstruct_iter: int
    _reconstruct_scale: float
    _reconstruct_spread: float
    _deflicker: bool
    _postprocess: bool
    _n_threads: int
    _has_previous: bool
    _has_current: bool
    _has_next: bool
    _morphology: Morphology

    __work_color: npt.NDArray[np.float64]
    __work_image: npt.NDArray[np.float64]
    __work_histogram: npt.NDArray[np.float64]
    __result: npt.NDArray[np.float32]
    __result_previous: npt.NDArray[np.float32]
    __result_current: npt.NDArray[np.float32]
    __result_next: npt.NDArray[np.float32]
    __deflicker_work: npt.NDArray[np.float32]

    def __init__(
        self,
        width: int,
        height: int,
        method: int = method_hybrid,
        n_iter: int = 4,
        sigma: float = 2.5,
        boundary_size: int | Literal['auto'] = 'auto',
        reconstruct: bool = True,
        reconstruct_iter: int = 10,
        reconstruct_scale: float = 0.5,
        reconstruct_spread: float = 0.025,
        deflicker: bool = True,
        postprocess: bool = True,
        n_threads: int = 0
    ) -> None:
        self._method = method
        self._n_iter = n_iter

        self._init_dimensions(width, height)
        self._init_dosage_properties(sigma, boundary_size)
        self._init_morphology(
            reconstruct,
            reconstruct_iter,
            reconstruct_scale,
            reconstruct_spread
        )

        self._deflicker = deflicker
        self._postprocess = postprocess
        self._n_threads = n_threads
        self.__allocate_resources()

    def _init_dimensions(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._area = width * height
        self._shape_hw = (height, width)
        self._shape_wh = (width, height)

    def _init_dosage_properties(
        self,
        sigma: float,
        boundary_size: int | Literal['auto'] = 'auto'
    ) -> None:
        self._sigma = sigma
        if boundary_size == 'auto':
            min_side = min(self._width, self._height)
            self._boundary_size = max(1, round(min_side * 0.1))
        else:
            self._boundary_size = boundary_size

    def _init_morphology(
        self,
        reconstruct: bool,
        reconstruct_iter: int,
        reconstruct_scale: float,
        reconstruct_spread: float
    ) -> None:
        self._reconstruct = reconstruct
        self._reconstruct_iter = reconstruct_iter
        self._reconstruct_scale = reconstruct_scale
        self._reconstruct_spread = reconstruct_spread

        if reconstruct:
            rw = round(self._width * self._reconstruct_scale)
            rh = round(self._height * self._reconstruct_scale)
            self._morphology = Morphology(rw, rh)

    def __allocate_resources(self) -> None:
        self.__work_color = np.zeros((self._area * 3), dtype=np.float64)
        self.__work_image = np.zeros((self._area * 13), dtype=np.float64)
        self.__work_histogram = np.zeros(((32 ** 3) * 8), dtype=np.float64)
        self.__result = np.zeros(self._shape_hw, dtype=np.float32)

        if self._deflicker:
            self._has_previous = False
            self._has_current = False
            self._has_next = False

            self.__result_previous = self.__result.copy()
            self.__result_current = self.__result.copy()
            self.__result_next = self.__result.copy()

            self.__deflicker_work = np.zeros(
                (3, self._height, self._width),
                dtype=np.float32
            )

    def _get_deflickered(self):
        prev = self.__result_previous
        curr = self.__result_current
        next_ = self.__result_next
        delta_p = np.abs(curr - prev)
        delta_n = np.abs(curr - next_)
        delta_p /= max(np.max(delta_p), 1e-8)
        delta_n /= max(np.max(delta_n), 1e-8)
        weight_p = (1 - delta_p)
        weight_n = (1 - delta_n)
        self.__result[:] = (curr + prev * weight_p + next_ * weight_n ) / (1 + weight_p + weight_n)

    def _do_deflicker(self) -> None:
        prev = self.__result_previous
        curr = self.__result_current
        next_ = self.__result_next

        work_prev = self.__deflicker_work[0]
        work_next = self.__deflicker_work[1]
        weight = self.__deflicker_work[2]

        delta_prev = work_prev
        delta_next = work_next

        np.subtract(curr, prev, out=delta_prev)
        np.abs(delta_prev, out=delta_prev)
        max_previous = max(np.max(delta_prev), 1e-8)
        np.divide(delta_prev, max_previous, out=delta_prev)

        np.subtract(curr, next_, out=delta_next)
        np.abs(delta_next, out=delta_next)
        max_next = max(np.max(delta_next), 1e-8)
        np.divide(delta_next, max_next, out=delta_next)

        weight_prev = work_prev
        weight_next = work_next

        np.subtract(1, delta_prev, out=weight_prev)
        np.subtract(1, delta_next, out=weight_next)

        weight[:] = 1
        np.add(weight, weight_prev, out=weight)
        np.add(weight, weight_next, out=weight)

        weighted_prev = work_prev
        weighted_next = work_next

        np.multiply(prev, weight_prev, out=weighted_prev)
        np.multiply(next_, weight_next, out=weighted_next)

        np.add(curr, weighted_prev, out=self.__result)
        np.add(self.__result, weighted_next, out=self.__result)
        np.divide(self.__result, weight, out=self.__result)

    def _get_reconstruct_selem(
        self,
        mask: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        total = np.sum(mask)
        area = np.sqrt(total)
        ks = int(self._reconstruct_spread * area)

        s = max(2, ks)
        if s % 2 == 0:
            s += 1

        return cv2.getStructuringElement(cv2.MORPH_RECT, (s, s))

    def _do_reconstruct(self) -> None:
        w = self._width
        h = self._height
        mw = self._morphology.get_width()
        mh = self._morphology.get_height()
        resize = w != mw or h != mh

        mask = self._morphology.mask

        if resize:
            cv2.resize(
                self.__result,
                (mw, mh),
                mask
            )

        else:
            mask[:] = self.__result

        niter = self._reconstruct_iter
        selem = self._get_reconstruct_selem(mask)
        cv2.erode(mask, selem, self._morphology.marker)
        result = self._morphology.reconstruct(Dilate, niter)
        self._morphology.mask[:] = result
        cv2.dilate(result, selem, self._morphology.marker)
        result = self._morphology.reconstruct(Erode, niter)

        if resize:
            cv2.resize(
                result,
                self._shape_wh,
                self.__result
            )

    def run(self, source: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32] | None:
        success = detect(
            self._width,
            self._height,
            np.reshape(source, (-1, 3), copy=False),
            self.__work_color,
            self.__work_image,
            self.__work_histogram,
            self._method,
            self._n_iter,
            self._sigma,
            self._boundary_size,
            self._n_threads,
        )

        if not success:
            return None

        result = self.__result

        result[:] = (
            self.__work_image[0:self._area]
            .reshape(self._shape_hw)
        )

        if self._reconstruct:
            self._do_reconstruct()

        if self._deflicker:
            if not self._has_previous:
                self._has_previous = True
                self.__result_previous[:] = result

            elif not self._has_current:
                self._has_current = True
                self.__result_current[:] = result

            elif not self._has_next:
                self._has_next = True
                self.__result_next[:] = result
                self._do_deflicker()

            else:
                previous = self.__result_previous
                previous[:] = result

                self.__result_previous = self.__result_current
                self.__result_current = self.__result_next
                self.__result_next = previous
                self._do_deflicker()

        if self._postprocess:
            np.subtract(result, 0.5, out=result)
            np.multiply(result, -10, out=result)
            np.exp(result, out=result)
            np.add(result, 1, out=result)
            np.reciprocal(result, out=result)

        return result.copy()


__all__ = ["Dosage"]
