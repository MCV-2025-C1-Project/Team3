"""
Collection of descriptors of an image
"""

import numpy as np
import cv2
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_histogram(hist: NDArray[np.float32], name_of_the_set: str, histogram_name: str, image_number: int) -> None:

    """
    Visualizes the histogram of the image
    
    Parameters
    ----------
        hist : NDArray
            Histogram of the image
        name_of_the_set : str
            Name of the set of the image
        histogram_name : str
            Name of the histogram
        image_number : int
            Number of the image
    """
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(hist)), hist, width=1.0, color="black")
    plt.xlabel("Bins (0-255)")
    plt.ylabel("Frequency")
    plt.title("Histogram")
    
    #Make sure results folder exists, otherwise create it
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{name_of_the_set}").mkdir(exist_ok=True)
    Path(f"results/{name_of_the_set}/{histogram_name}").mkdir(exist_ok=True)

    plt.savefig(f"results/{name_of_the_set}/{histogram_name}/{image_number:05d}.png", dpi=300, bbox_inches="tight")
    plt.close()


def compute_histogram(img: NDArray[np.uint8]) -> NDArray[np.float32]:
    
    """
    Computes the histogram of the image
    
    Parameters
    ----------
        img : NDArray
            Image to compute the histogram from
    -------
        hist : array-like
            Histogram of the given image
    """
    
    img = img.astype(int)

    hist = [0] * 256
    hist = np.array(hist, dtype=np.float32)
    
    unique, counts = np.unique(img, return_counts=True)
    hist[unique] = counts
    
    hist /= (hist.sum() + 1e-7)
    return hist

def rgb_descriptor(img : NDArray, name_of_the_set: str, image_number: int, visualize: bool = False) -> NDArray:
    """
    Computes the concatenation of the RGB histograms
    
    Parameters
    ----------
        img : NDArray
            An RGB image
        name_of_the_set : str
            Name of the set of the image
        image_number : int
            Number of the image
        visualize : bool, optional (default=False)
            If True, plots and saves the histogram 
    Returns
    -------
        hist : NDArray
            Concatenated RGB histogram of the image
    """
    r = compute_histogram(img[0])
    g = compute_histogram(img[1])
    b = compute_histogram(img[2])
    rgb_hist = np.concat([r,g,b])
    if visualize:
        visualize_histogram(rgb_hist, name_of_the_set, "rgb_histogram", image_number)
    return rgb_hist

def gray_descriptor(img : NDArray, name_of_the_set: str, image_number: int, visualize: bool = False) -> NDArray:
    """
    Computes the gray level histogram of an image
    
    Parameters
    ----------
        img : NDArray
            An RGB image
        name_of_the_set : str
            Name of the set of the image
        image_number : int
            Number of the image
        visualize : bool, optional (default=False)
            If True, plots and saves the histogram
    Returns
    -------
        hist : NDArray
            Gray level histogram the image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_hist = compute_histogram(img)
    if visualize:
        visualize_histogram(gray_hist,  name_of_the_set,"gray_histogram",image_number)
    return gray_hist

def hsv_descriptor(img : NDArray, name_of_the_set: str, image_number: int, visualize: bool = False) -> NDArray:
    """
    Computes the concatenation of the HSV histogram
    
    Parameters
    ----------
        img : NDArray
            An RGB image
        visualize : bool, optional (default=False)
            If True, plots and saves the histogram
        name_of_the_set : str
            Name of the set of the image
        image_number : int
            Number of the image
    Returns
    -------
        hist : array-like
            HSV histogram of the image
    """
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv_image)

    h_hist = compute_histogram(h)
    s_hist = compute_histogram(s)
    v_hist = compute_histogram(v)
    hsv_hist = np.concat([h_hist, s_hist, v_hist])
    if visualize:
        visualize_histogram(hsv_hist, name_of_the_set, "hsv_histogram", image_number)
    return hsv_hist



if __name__ == "__main__":
    
    image_path = "data/BBDD/bbdd_00000.jpg"
    img = cv2.imread(image_path)
    print(f"Red Hist:{compute_histogram(img[0], save_img=True, save_name="Red histogram")}")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(f"Gray Hist:{compute_histogram(img, save_img=True, save_name="Gray histogram")}")