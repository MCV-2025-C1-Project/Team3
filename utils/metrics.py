"""
Collection of distance metrics for feature vectors
"""

import numpy as np
from numpy.typing import NDArray
    
def euclidean_distance(x: NDArray, y: NDArray) -> float:
    """
    Compute the Euclidean distance (L2 distance) between two NumPy arrays
    
    Parameters
    ----------
        x : np.ndarray
            First array
        y : np.ndarray
            Second array
    Returns
    ----------
        out : float
            The Euclidean distance between vectors x and y
    """
    return np.sqrt(np.sum((x - y) ** 2))


def l1_distance(x: NDArray, y: NDArray) -> float:
    """
    Compute the L1 distance between two NumPy arrays

    Parameters
    ----------
        x : np.ndarray
            First array
        y : np.ndarray
            Second array
    Returns
    ----------
        out : float
            The L1 distance between vectors x and y

    """
    return np.sum(np.abs(x - y))


def x2_dist(x: NDArray, y: NDArray) -> float:
    """
    Compute the X squared distance between two NumPy arrays

    Parameters
    ----------
        x : 1-D np.ndarray
            First array
        y : np.ndarray
            Second array
    Returns
    ----------
        out : float
            The X squared distance between vectors x and y

    """

    numerator = (x - y) ** 2
    denominator = x + y

    # For avoiding division by zero
    denominator[denominator == 0] = 1e-10

    return np.sum(numerator / denominator)


def histogram_intersection(hist1: NDArray, hist2: NDArray) -> float:
    """
    Compute the histogram intersection similarity
    
    Parameters
    ----------
        hist1 : 1-D np.ndarray
            First histogram
        hist2 : 1-D np.ndarray
            Second histogram

    Returns
    -------
        out : float
            Histogram intersection similarity between the two given histograms
    """

    return np.sum(np.minimum(hist1, hist2))


def hellinger_kernel(x: NDArray, y: NDArray) -> float:
    """
    Compute the Hellinger kernel similarity
    
    Parameters
    ----------
        x : np.ndarray
            First array
        y : np.ndarray
            Second array

    Returns
    -------
        out : float
            Hellinger kernel similarity between the two given NumPy arrays
    """

    return np.sum(np.sqrt(x * y))


def bhattacharyya_distance(x: NDArray, y: NDArray) -> float:
    """
    Compute the Bhattacharyya distance
    
    Parameters
    ----------
        x : np.ndarray
            First array
        y : 1-D np.ndarray
            Second array

    Returns
    -------
        out : float
            Bhattacharyya distance between the two given NumPy arrays
    """
    
    return -np.log(np.sum(np.sqrt(x * y)) + 1e-10)

def average_precision_k(ground_truth : list, predicted : list, k : int) -> float:
    """
    Compute the average precision at K
    
    Parameters
    ----------
        ground truth : list
            A list of the ground truth items
        predicted : list
            A list of the predicted items
        k : int
            The maximum number of predictions to consider

    Returns
    -------
        out : float
            Average precision at k
    """
    
    score = 0.0
    num_hits = 0.0
    
    for i, p in enumerate(predicted[:k]):
        if p in ground_truth:
            num_hits += 1
            score += num_hits / (i + 1.0)
            
    if num_hits > 0:
        return score / num_hits
    return 0

if __name__ == "__main__":
    vec1 = np.array([1.0, 2.0, 3.0, 4.0])
    vec2 = np.array([2.0, 2.0, 2.0, 2.0])
    print(f"Euclidean distance: {euclidean_distance(vec1, vec2)}")
    print(f"L1 distance: {l1_distance(vec1, vec2)}")
    print(f"X2 distance: {x2_dist(vec1, vec2)}")
    print(f"Histogram intersection: {histogram_intersection(vec1, vec2)}")
    print(f"Hellinger kernel: {hellinger_kernel(vec1, vec2)}")
    print(f"Bhattacharyya distance: {bhattacharyya_distance(vec1, vec2)}")
