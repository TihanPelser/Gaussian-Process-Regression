from typing import Any
import numpy as np
from scipy.spatial.distance import cdist
from abc import ABC, abstractmethod


class Kernel(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)


class RBF(ABC):
    def __init__(self, sigma: float = 2.2, l: float = 0.5) -> None:
        """Instantiate kernel

        Args:
            sigma (float, optional): Height scale of the kernel. Defaults to 2.2.
            l (float, optional): Length scale of the kernel. Defaults to 0.5.
        """
        super().__init__()
        self.sigma = sigma
        self.l = l

    def __call__(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        """Evualate the RBF kernel.

        Args:
            xi (np.ndarray): Xi
            xj (np.ndarray): Xj

        Returns:
            np.ndarray: Covariance matrix (nxn), cov(x_i, x_j)
        """
        d = cdist(xi/self.l, xj/self.l, metric="sqeuclidean")

        return self.sigma**2 * np.exp(-0.5*d)
    

class Periodic(ABC):
    def __init__(self, sigma: float = 2.2, l: float = 0.5, p: float = 0.5) -> None:
        """Instantiate kernel

        Args:
            sigma (float, optional): Height scale of the kernel. Defaults to 2.2.
            l (float, optional): Length scale of the kernel. Defaults to 0.5.
            p (float, optional): Period of the kernel. Defaults to 0.5.
        """
        super().__init__()
        self.sigma = sigma
        self.l = l
        self.p = p

    def __call__(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        """Evualate the Periodic kernel.

        Args:
            xi (np.ndarray): Xi
            xj (np.ndarray): Xj

        Returns:
            np.ndarray: Covariance matrix (nxn), cov(x_i, x_j)
        """
        d = cdist(xi, xj, metric="euclidean")

        exponential = (np.sin(np.pi*d/self.p)/self.l)**2

        return self.sigma**2 * np.exp(-2*exponential)
