import numpy as np
import scipy.stats as ss
from .kernels import Kernel
from scipy.linalg import solve_triangular, cho_factor, cho_solve
from typing import Tuple, Union


class GPR:
    def __init__(self, kernel: Kernel, beta: float = 1e-5) -> None:
        """GP class

        Args:
            kernel (Kernel): The kernel (or covariance function) of the GP
            beta (float, optional): A small value to add to the diagonal of the covariance matrix to prevent 
            numerical issues (covariance matrix must be postive definite). Defaults to 1e-5.
        """
        self.kernel = kernel
        self.beta = beta
        self._K = None
        self._L = None
        self._X = None
        self._y = None
        self._sigma_n = None
        self._alpha = None


    def fit(self, X: np.ndarray, y: np.ndarray, sigma_n: float) -> None:
        """'Fit' the GP. This only precomputes some of the variables and matrices needed for the prediction or regression step.

        Args:
            X (np.ndarray): Training/observed set locations 
            y (np.ndarray): Training/observed set values
            sigma_n (float): The standard deviation (scale) of the assumed additive Gaussian measurement noise
        """
        
        self._X = X
        self._y = y
        self._sigma_n = sigma_n
       
        self._K = self.kernel(xi=X, xj=X)
        
        self._K[np.diag_indices_from(self._K)] += self.beta  # Add constant to prevent singular matrix issue

        # Lower triangular (Cholesky decomposition)
        self._L, _ = cho_factor(self._K + self._sigma_n**2 * np.eye(self._X.size), lower=True)

        self._alpha = cho_solve((self._L, True), y, check_finite=True)


    def predict(self, Xs: np.ndarray, return_cov: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Perform GP regression and return the mean function (f) of the posterior distribution. 
        If return_cov is set to True, the covariance will also be returned. 

        Args:
            Xs (np.ndarray): The points at which to evaluate the mean. The test points.
            return_cov (bool, optional): Whether to calculate and return the covariance matrix. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], np.ndarray]: _description_
        """

        Ks = self.kernel(xi=Xs, xj=self._X)

        Fs = np.dot(Ks, self._alpha)

        if not return_cov:
            return Fs

        v = solve_triangular(self._L, Ks.T, lower=True)

        Kss = self.kernel(xi=Xs, xj=Xs)

        V = Kss - np.dot(v.T, v)

        return Fs, V
    

    def sample(self, Xs: np.ndarray, n: int = 1) -> np.ndarray:
        """Sample from the GP.

        Args:
            Xs (np.ndarray): Points at which to sample from the GP
            n (int, optional): Number of samples to take at each x-point. Defaults to 1.

        Returns:
            (np.ndarray): The samples in the shape (n_samples, n_points)
        """
        
        Fs, cov = self.predict(Xs, return_cov=True)

        cov[np.diag_indices_from(cov)] += self.beta  # Add constant to prevent singular matrix issue

        return ss.multivariate_normal(mean=Fs.flatten(), cov=cov).rvs(n)
