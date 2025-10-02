"""
Collection of descriptors of an image
"""

import numpy as np
import cv2
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_histogram(hist: NDArray[np.float32], name_of_the_set: str, histogram_name: str, image_number: int, channel_labels: list[str] = None,channel_sizes: list[int] = None) -> None:
    """
    Plot and save a histogram visualization for an image.

    Parameters
    ----------
    - hist : NDArray[np.float32]
        Concatenated histogram values (e.g., RGB, HSV, Gray).
    - name_of_the_set : str
        Dataset name (e.g., "BBDD", "qsd1_w1").
    - histogram_name : str
        Identifier of the histogram type.
    - image_number : int
        Image index (used for naming the file).
    - channel_labels : list of str, optional
        Labels for each channel (["R","G","B"], ["H","S","V"], ["Gray"]).
    - channel_sizes : list of int, optional
        Number of bins for each channel. If None, assumes equal length.
    """

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(hist)), hist, width=1.0, color="black")
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.title(histogram_name)

    if channel_labels and channel_sizes:
        positions = []
        labels = []
        offset = 0
        for size, label in zip(channel_sizes, channel_labels):
            positions.append(offset + size // 2)
            labels.append(label)
            # Add vertical line (except at start)
            if offset > 0:
                plt.axvline(x=offset, color="red", linestyle="--", linewidth=1)
            positions.append(offset + size - 1)
            labels.append(str(size - 1))

            offset += size

        plt.xticks(positions, labels, rotation=45)

    Path(f"results/histograms/{name_of_the_set}/{histogram_name}").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"results/histograms/{name_of_the_set}/{histogram_name}/{image_number:05d}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

def compute_histogram(img: NDArray[np.uint8], bins: int = 256, value_range: tuple[int,int] = (0,256)) -> NDArray[np.float32]:
    """
    Compute a normalized histogram for a single-channel image.

    Parameters
    ----------
    - img : NDArray[np.uint8]
        Input single-channel image.
    - bins : int
        Number of bins in the histogram.
    - value_range : tuple of int
        The (min, max) value range for the histogram.

    Returns
    -------
    - hist : NDArray[np.float32]
        Normalized histogram of shape (bins,).
    """
    hist, _ = np.histogram(img, bins=bins, range=value_range)
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    return hist

def rgb_descriptor(img : NDArray, name_of_the_set: str, image_number: int, visualize: bool = False) -> NDArray:
    """
    Computes the concatenation of the RGB histograms
    
    Parameters
    ----------
        - img : NDArray
            An RGB image
        - name_of_the_set : str
            Name of the set of the image
        - image_number : int
            Number of the image
        - visualize : bool, optional (default=False)
            If True, plots and saves the histogram 
    Returns
    -------
        - rgb_hist : NDArray
            Concatenated RGB histogram of the image
    """
    r = compute_histogram(img[0],bins=256, value_range=(0,256))
    g = compute_histogram(img[1],bins=256, value_range=(0,256))
    b = compute_histogram(img[2],bins=256, value_range=(0,256))
    rgb_hist = np.concat([r,g,b])

    if visualize:
        visualize_histogram(rgb_hist, name_of_the_set, "rgb_histogram", image_number,channel_labels=["R", "G", "B"],channel_sizes=[256,256,256])
    return rgb_hist

def gray_descriptor(img : NDArray, name_of_the_set: str, image_number: int, visualize: bool = False) -> NDArray:
    """
    Computes the gray level histogram of an image
    
    Parameters
    ----------
        - img : NDArray
            An RGB image
        - name_of_the_set : str
            Name of the set of the image
        - image_number : int
            Number of the image
        - visualize : bool, optional (default=False)
            If True, plots and saves the histogram
    Returns
    -------
        - gray_hist : NDArray
            Gray level histogram the image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_hist = compute_histogram(img, bins=256, value_range=(0,256))

    if visualize:
        visualize_histogram(gray_hist,  name_of_the_set,"gray_histogram",image_number,channel_labels=["Gray"],channel_sizes=[256])
    return gray_hist

def hsv_descriptor(img : NDArray, name_of_the_set: str, image_number: int, visualize: bool = False) -> NDArray:
    """
    Computes the concatenation of the HSV histogram
    
    Parameters
    ----------
        - img : NDArray
            An RGB image
        - visualize : bool, optional (default=False)
            If True, plots and saves the histogram
        - name_of_the_set : str
            Name of the set of the image
        - image_number : int
            Number of the image
    Returns
    -------
        - hsv_hist : array-like
            HSV histogram of the image
    """
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv_image)

    h_hist = compute_histogram(h, bins=180, value_range=(0,180))
    s_hist = compute_histogram(s,bins=256, value_range=(0,256))
    v_hist = compute_histogram(v,bins=256, value_range=(0,256))
    hsv_hist = np.concat([h_hist, s_hist, v_hist])

    if visualize:
        visualize_histogram(hsv_hist, name_of_the_set, "hsv_histogram", image_number,channel_labels=["H", "S", "V"],channel_sizes=[180,256,256])
    return hsv_hist