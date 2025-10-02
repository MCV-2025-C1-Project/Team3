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
    Computes the concatenation of the BGR histograms
    
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

    b,g,r = cv2.split(img)

    b_hist = compute_histogram(b, bins=256, value_range=(0,256))
    g_hist = compute_histogram(g,bins=256, value_range=(0,256))
    r_hist = compute_histogram(r,bins=256, value_range=(0,256))
    
    bgr_hist = np.concat([b_hist, g_hist, r_hist])

    if visualize:
        visualize_histogram(bgr_hist, name_of_the_set, "rgb_histogram", image_number,channel_labels=["B", "G", "R"],channel_sizes=[256,256,256])
    return bgr_hist

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


def lab_descriptor(img: NDArray, name_of_the_set: str, image_number: int, visualize: bool = False) -> NDArray:
    """
    Computes the concatenation of the CIELAB histogram
    
    Parameters
    ----------
        img : NDArray
            A BGR image
        visualize : bool, optional (default=False)
            If True, plots and saves the histogram
        name_of_the_set : str
            Name of the set of the image
        image_number : int
            Number of the image
    Returns
    -------
        hist : array-like
            Lab histogram of the image
    """
    
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab_image)

    l_hist = compute_histogram(l,bins=256, value_range=(0,256))
    a_hist = compute_histogram(a,bins=256, value_range=(0,256))
    b_hist = compute_histogram(b,bins=256, value_range=(0,256))

    lab_hist = np.concatenate([l_hist, a_hist, b_hist])

    if visualize:
        visualize_histogram(lab_hist, name_of_the_set, "lab_histogram", image_number, channel_labels=["L", "A", "B"],channel_sizes=[256,256,256])

    return lab_hist


def ycbcr_descriptor(img: NDArray, name_of_the_set: str, image_number: int, visualize: bool = False) -> NDArray:
    """
    Computes the concatenation of the Y, Cb, Cr histograms from an image.

    Parameters
    ----------
    img : NDArray
        Input BGR image
    name_of_the_set : str
        Name of the dataset the image belongs to
    image_number : int
        Index/ID of the image
    visualize : bool, optional (default=False)
        If True, plots and saves the histogram

    Returns
    -------
    hist : NDArray
        Concatenated YCbCr histogram of the image
    """
    ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_image)

    y_hist  = compute_histogram(y,bins=256, value_range=(0,256))
    cb_hist = compute_histogram(cb,bins=256, value_range=(0,256))
    cr_hist = compute_histogram(cr,bins=256, value_range=(0,256))

    ycbcr_hist = np.concatenate([y_hist, cb_hist, cr_hist])

    if visualize:
        visualize_histogram(ycbcr_hist, name_of_the_set, "ycbcr_histogram", image_number, channel_labels=["Y", "Cb", "Cr"],channel_sizes=[256,256,256])

    return ycbcr_hist


def mix_of_all_descriptor(img: NDArray, name_of_the_set: str, image_number: int, visualize: bool = False) -> NDArray:
    """
    Computes the concatenation of the Grayscale, BGR, HSV, CIELAB, and YCbCr histograms from an image.

    Parameters
    ----------
    img : NDArray
        Input BGR image
    name_of_the_set : str
        Name of the dataset the image belongs to
    image_number : int
        Index/ID of the image
    visualize : bool, optional (default=False)
        If True, plots and saves the histogram

    Returns
    -------
    hist : NDArray
        Concatenated 1-D histogram of the color spaces mentioned above for the input image
    """

    bgr_hist = rgb_descriptor(img, name_of_the_set, image_number, visualize=False)
    gray_hist = gray_descriptor(img, name_of_the_set, image_number, visualize=False)
    hsv_hist = hsv_descriptor(img, name_of_the_set, image_number, visualize=False)
    lab_hist = lab_descriptor(img, name_of_the_set, image_number, visualize=False)
    ycbcr_hist = ycbcr_descriptor(img, name_of_the_set, image_number, visualize=False)

    mix_of_all_hist = np.concatenate([bgr_hist, gray_hist, hsv_hist, lab_hist, ycbcr_hist])

    if visualize:
        visualize_histogram(mix_of_all_hist, name_of_the_set, "mix_of_all_histogram", image_number, channel_labels=["BGR", "Gray", "HSV", "CIELAB", "YCbCr"],channel_sizes=[768,256,692,768,768])

    return mix_of_all_hist