"""
Precomputes the descriptors for all the DDBB images.    
"""

"""

To run this without issues because of python not considering it part of the package,
run it from the Team3 folder using python -m descriptors/descriptor_creator.py

"""

import cv2
import utils.descriptors as descriptors
import numpy as np

def rgb_descriptor(image_path : str) -> np.ndarray:
    img = cv2.imread(image_path)
    r = descriptors.compute_histogram(img[0])
    g = descriptors.compute_histogram(img[1])
    b = descriptors.compute_histogram(img[2])
    rgb_hist = np.concat([r,g,b])
    return rgb_hist

def gray_descriptor(image_path : str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray_hist = descriptors.compute_histogram(img[0])
    return gray_hist

NUMBER_OF_FILES = 287
RESULT_FILENAME = "gray_example.txt"

with open(f"descriptors/{RESULT_FILENAME}", "w") as f:
    for i in range(NUMBER_OF_FILES):
        image_path = f"data/BBDD/bbdd_{i:05d}.jpg"
        descriptor = gray_descriptor(image_path)
        np.savetxt(f, descriptor[None])
    