"""
    Collection of descriptors of an image
"""

import numpy as np
import cv2
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from pathlib import Path

## Discuss usage of numpy
def compute_histogram(img: NDArray[np.uint8], save_img: bool = False, save_name : str = "histogram") -> NDArray[np.float32]:
    
    """
    Computes the histogram of the image
    
    Parameters
    ----------
        img : MatLike
            Image to compute the histogram from
        save_img : bool
            If set to `True` saves the image on `results/{save_name}.png`
        save_name : str
            If save_img is `True` it sets the name of the saved image
    Returns
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


    if save_img:
        
        plt.figure(figsize=(12, 4))
        plt.bar(range(256), hist, width=1.0, color="black")
        plt.xlabel("Bins (0-255)")
        plt.ylabel("Frequency")
        plt.title("Histogram")
        
        #Make sure results folder exists, otherwise create it
        Path("results").mkdir(exist_ok=True)
        
        plt.savefig(f"results/{save_name}.png", dpi=300, bbox_inches="tight")

    return hist

def rgb_descriptor(img : NDArray) -> NDArray:
    """
    Computes the concatenation of the RGB histograms
    
    Parameters
    ----------
        img : MatLike
            An RGB image
    Returns
    -------
        hist : array-like
            Concatenated RGB histogram of the image
    """
    r = compute_histogram(img[0])
    g = compute_histogram(img[1])
    b = compute_histogram(img[2])
    rgb_hist = np.concat([r,g,b])
    return rgb_hist

def gray_descriptor(img : NDArray) -> NDArray:
    """
    Computes the gray level histogram of an image
    
    Parameters
    ----------
        image_path : str
            An RGB image
    Returns
    -------
        hist : array-like
            Gray level histogram the image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_hist = compute_histogram(img)
    return gray_hist


if __name__ == "__main__":
    
    image_path = "data/BBDD/bbdd_00000.jpg"
    img = cv2.imread(image_path)
    print(f"Red Hist:{compute_histogram(img[0], save_img=True, save_name="Red histogram")}")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(f"Gray Hist:{compute_histogram(img, save_img=True, save_name="Gray histogram")}")