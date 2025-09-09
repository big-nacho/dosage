import numpy as np
import numpy.typing as npt
from typing import Union, Literal
from .Morphology import Morphology
from .pydosage import method_hybrid, boundary_none

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
    ) -> None : ...

    def _init_dimensions(self, width: int, height: int) -> None: ...

    def _init_dosage_properties(
        self,
        sigma: float,
        boundary_thickness: Union[int, Literal['auto']]
    ) -> None: ...

    def _init_morphology(self) -> None: ...

    def __allocate_resources(self) -> None: ...

    def _get_reconstruct_selem(
        self,
        mask: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]: ...

    def _do_reconstruct(self) -> None: ...

    def _do_deflicker(self) -> None: ...

    def _normalize(self) -> None: ...

    def run(self, source: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32] | None: ...



