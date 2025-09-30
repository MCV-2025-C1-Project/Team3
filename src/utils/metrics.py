"""
metrics.py

This module defines a collection of distance metrics.
Implemented metrics:
- l1_distance


"""

from typing import List
import numpy as np
from numpy.typing import NDArray

class Metrics:
    """
    Collection of distance metrics for feature vectors
    """

    @staticmethod
    def l1_distance(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        """
        Compute the L1 distance between two NumPy arrays

        Args:
            - x (NDArray[np.float64]): First vector
            - y (NDArray[np.float64]): Second vector
        
        Returns:
            float: The L1 distance between vectors x and y

        """
        return np.sum(np.abs(x - y))
    
metrics = Metrics()
print(metrics.l1_distance(np.array([1,2,3,4]),np.array([2,2,2,2])))