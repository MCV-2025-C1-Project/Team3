import numpy as np
from numpy.typing import NDArray


class Metrics:
    """
    Collection of distance metrics for feature vectors
    """

    @staticmethod
    def euclidean_distance(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        """
        Compute the Euclidean distance (L2 distance) between two NumPy arrays
        Args:
            - x (NDArray[np.float64]): First vector
            - y (NDArray[np.float64]): Second vector
        Returns:
            float: The Euclidean distance between vectors x and y
        """
        return np.sqrt(np.sum((x - y) ** 2))

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

        # For avoiding division by zero
        denominator[denominator == 0] = 1e-10

        return np.sum(numerator / denominator)

    @staticmethod
    def histogram_intersection(
        hist1: NDArray[np.float64], hist2: NDArray[np.float64]
    ) -> float:
        """
        Input:
        -hist1 (1-D numpy array): First image descriptor
        -hist2 (1-D numpy array): Second image descriptor

        Returns:
        -total (float): Histogram intersection similarity between the two descriptors
        """

        return np.sum(np.minimum(hist1, hist2))

    @staticmethod
    def hellinger_kernel(
        hist1: NDArray[np.float64], hist2: NDArray[np.float64]
    ) -> float:
        """
        Input:
        -hist1 (1-D numpy array): First image descriptor
        -hist2 (1-D numpy array): Second image descriptor

        Returns:
        -total (float): Hellinger kernel similarity between the two descriptors
        """

        return np.sum(np.sqrt(hist1 * hist2))

    @staticmethod
    def bhattacharyya_distance(
        hist1: NDArray[np.float64], hist2: NDArray[np.float64]
    ) -> float:
        """
        Compute the Bhattacharyya distance between two histograms
        Input:
            -hist1 (1-D numpy array): First image descriptor
            -hist2 (1-D numpy array): Second image descriptor
        Returns:
            -total (float): Bhattacharyya distance between the two descriptors
        """
        return -np.log(np.sum(np.sqrt(hist1 * hist2)) + 1e-10)


metrics = Metrics()
vec1 = np.array([1.0, 2.0, 3.0, 4.0])
vec2 = np.array([2.0, 2.0, 2.0, 2.0])
print(f"Euclidean distance: {metrics.euclidean_distance(vec1, vec2)}")
print(f"L1 distance: {metrics.l1_distance(vec1, vec2)}")
print(f"X2 distance: {metrics.x2_dist(vec1, vec2)}")
print(f"Histogram intersection: {metrics.histogram_intersection(vec1, vec2)}")
print(f"Hellinger kernel: {metrics.hellinger_kernel(vec1, vec2)}")
print(f"Bhattacharyya distance: {metrics.bhattacharyya_distance(vec1, vec2)}")
