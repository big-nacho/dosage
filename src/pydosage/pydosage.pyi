import numpy as np
import numpy.typing as npt
from typing import Union, Literal

def detect(
    width: int,
    height: int,
    colors: npt.NDArray[np.uint8],
    work_color: npt.NDArray[np.float64],
    work_image: npt.NDArray[np.float64],
    work_histogram: npt.NDArray[np.float64],
    method: int,
    n_iter: int,
    sigma: float,
    boundary_thickness: Union[int | Literal["auto"]],
    n_threads: int
) -> bool: ...

method_dosage: int
method_fast_mbd: int
method_hybrid: int