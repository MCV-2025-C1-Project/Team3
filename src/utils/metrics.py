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

    
    @staticmethod
    def x2_dist(hist1: NDArray[np.float64], hist2: NDArray[np.float64]) -> float:
        """
        Input:
        -hist1 (1-D numpy array): First image descriptor
        -hist2 (1-D numpy array): Second image descriptor
        
        Returns:
        -total (float): X2 distance between the two descriptors
        """

        numerator = (hist1 - hist2) ** 2
        denominator = hist1 + hist2

        #For avoiding division by zero
        denominator[denominator == 0] = 1e-10

        return np.sum(numerator / denominator)
    

    @staticmethod
    def histogram_intersection(hist1: NDArray[np.float64], hist2: NDArray[np.float64]) -> float:
        """
        Input:
        -hist1 (1-D numpy array): First image descriptor
        -hist2 (1-D numpy array): Second image descriptor
        
        Returns:
        -total (float): Histogram intersection similarity between the two descriptors
        """

        return np.sum(np.minimum(hist1, hist2))
    
    @staticmethod
    def hellinger_kernel(hist1: NDArray[np.float64], hist2: NDArray[np.float64]) -> float:
        """
        Input:
        -hist1 (1-D numpy array): First image descriptor
        -hist2 (1-D numpy array): Second image descriptor
        
        Returns:
        -total (float): Hellinger kernel similarity between the two descriptors
        """

        return np.sum(np.sqrt(hist1*hist2))
    
metrics = Metrics()
print(metrics.l1_distance(np.array([1,2,3,4]),np.array([2,2,2,2])))
print(metrics.x2_dist(np.array([1,2,3,4]),np.array([2,2,2,2])))
print(metrics.histogram_intersection(np.array([1,2,3,4]),np.array([2,2,2,2])))
print(metrics.hellinger_kernel(np.array([1,2,3,4]),np.array([2,2,2,2])))

