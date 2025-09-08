import cv2
import numpy as np
from .Morphology import Morphology, Dilate, Erode
from typing import Literal
import numpy.typing as npt

from .dosage import (
    detect,
    method_dosage,
    method_fast_mbd,
    method_hybrid,
    boundary_top,
    boundary_right,
    boundary_bottom,
    boundary_left,
    boundary_none
)


class Dosage:

    MethodDosage: int = method_dosage
    MethodFastMBD: int = method_fast_mbd
    MethodHybrid: int = method_hybrid
    BoundaryTop: int = boundary_top
    BoundaryRight: int = boundary_right
    BoundaryBottom: int = boundary_bottom
    BoundaryLeft: int = boundary_left
    BoundaryNone: int = boundary_none

    _width: int
    _height: int
    _area: int
    _shape_hw: tuple[int, int]
    _shape_wh = tuple[int, int]
    _method: int
    _sigma: float
    _boundary_thickness: int | Literal['auto']
    _foreground_boundary: int
    _n_passes: int
    _reconstruct: bool
    _reconstruct_iter: int
    _reconstruct_scale: float
    _reconstruct_spread: float
    _deflicker: bool
    _postprocess: bool
    _sigmoid_center: float
    _sigmoid_strength: float
    _winsor: float
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
        sigma: float = 2.5,
        boundary_thickness: int | Literal['auto'] = 'auto',
        foreground_boundary: int = boundary_none,
        n_passes: int = 4,
        winsor: float = 0,
        reconstruct: bool = True,
        reconstruct_iter: int = 10,
        reconstruct_scale: float = 0.5,
        reconstruct_spread: float = 0.025,
        reconstruct_renormalize: bool = True,
        deflicker: bool = True,
        postprocess: bool = True,
        sigmoid_center: float = 0.5,
        sigmoid_strength: float = 10
    ) -> None:
        self._method = method
        self._foreground_boundary = foreground_boundary
        self._n_passes = n_passes

        self._init_dimensions(width, height)
        self._init_dosage_properties(sigma, boundary_thickness)

        self._reconstruct = reconstruct
        self._reconstruct_iter = reconstruct_iter
        self._reconstruct_scale = reconstruct_scale
        self._reconstruct_spread = reconstruct_spread
        self._reconstruct_renormalize = reconstruct_renormalize

        if reconstruct:
            self._init_morphology()

        self._winsor = winsor
        self._deflicker = deflicker
        self._postprocess = postprocess
        self._sigmoid_center = sigmoid_center
        self._sigmoid_strength = sigmoid_strength
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
        boundary_thickness: int | Literal['auto'] = 'auto'
    ) -> None:
        self._sigma = sigma
        if boundary_thickness == 'auto':
            min_side = min(self._width, self._height)
            self._boundary_thickness = max(1, round(min_side * 0.1))
        else:
            self._boundary_thickness = boundary_thickness

    def _init_morphology(self) -> None:
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

    def _do_winsor(self) -> None:
        np.clip(
            self.__result, 
            np.percentile(self.__result, self._winsor), 
            np.percentile(self.__result, 100 - self._winsor),
            out=self.__result
        )

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

        selem = self._get_reconstruct_selem(mask)
        cv2.erode(mask, selem, self._morphology.marker)
        result = self._morphology.reconstruct(Dilate, self._reconstruct_iter)
        self._morphology.mask[:] = result
        cv2.dilate(result, selem, self._morphology.marker)
        result = self._morphology.reconstruct(Erode, self._reconstruct_iter)

        if resize:
            cv2.resize(
                result,
                self._shape_wh,
                self.__result,
                interpolation=cv2.INTER_LANCZOS4
            )

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

    def _normalize(self) -> None:
        max = np.max(self.__result)
        if max > 0:
            np.divide(
                self.__result, 
                max, 
                out=self.__result
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
            self._sigma,
            self._boundary_thickness,
            self._foreground_boundary,
            self._n_passes,
        )

        if not success:
            return None

        result = self.__result

        result[:] = (
            self.__work_image[0:self._area]
            .reshape(self._shape_hw)
        )
        
        if self._winsor > 0:
            self._do_winsor()
            self._normalize()

        if self._reconstruct:
            self._do_reconstruct()
            if self._reconstruct_renormalize:
                self._normalize()

        if self._postprocess:
            np.subtract(result, self._sigmoid_center, out=result)
            np.multiply(result, -self._sigmoid_strength, out=result)
            np.exp(result, out=result)
            np.add(result, 1, out=result)
            np.reciprocal(result, out=result)

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

        return result.copy()


__all__ = ["Dosage"]
