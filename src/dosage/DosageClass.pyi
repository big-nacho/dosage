import numpy as np
import numpy.typing as npt
from typing import Union, Literal, Optional
from .Morphology import Morphology
from .dosage import method_hybrid

class Dosage:
    Dosage: int
    Fast_MBD: int
    Hybrid: int

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
    ) -> None : ...

    def _init_dimensions(self, width: int, height: int) -> None: ...

    def _init_dosage_properties(
        self,
        sigma: float,
        boundary_size: Union[int, Literal['auto']]
    ) -> None: ...

    def _init_morphology(
        self,
        reconstruct: bool,
        reconstruct_iter: int,
        reconstruct_scale: float,
        reconstruct_spread: float
    ) -> None: ...

    def __allocate_resources(self) -> None: ...

    def _do_deflicker(self) -> None: ...

    def _get_reconstruct_selem(
        self,
        mask: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]: ...

    def _do_reconstruct(self) -> None: ...

    def run(self, source: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32] | None: ...



