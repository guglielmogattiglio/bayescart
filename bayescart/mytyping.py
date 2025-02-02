from typing import Sequence, TypeVar, Any, Iterable
import numpy.typing as npt
import numpy as np

T = TypeVar('T', str, int, float)
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float64]