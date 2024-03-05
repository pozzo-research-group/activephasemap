from botorch.test_functions import SyntheticTestFunction
import torch 
from math import pi 
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from torch import Tensor

class SinX(SyntheticTestFunction):
    r"""SinX test function.

    1-dimensional function (evaluated on [-3*pi, 3*pi]):

        f(x) = sin(x)

    """
    _check_grad_at_opt: bool = False

    def __init__(
        self,
        dim: int = 1,
        noise_std: Optional[float] = 0.3,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = dim
        if bounds is None:
            bounds = [(-1.0, 2.0) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return -torch.sin(3*X) - X**2 + 0.7*X