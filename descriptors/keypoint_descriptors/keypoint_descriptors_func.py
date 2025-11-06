"""
Collection of descriptors of an image
"""

import numpy as np
from config import io_config, general_config
from background_removal.main_background_removal import main_background_removal
import cv2
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.feature import local_binary_pattern
from scipy.fftpack import dctn
from skimage.util import view_as_blocks
import pywt


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

    output_dir = io_config.HIST_DIR / name_of_the_set / histogram_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{image_number:05d}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.close()



def generic_keypoint_descriptor(descriptor_type: str,
                       descriptor_specific_parameters: dict,):
    """
    Factory function that creates a descriptor function from a given configuration.

    Parameters
    ----------
    color_space : str
        Texture descriptor typw (e.g., "LBP").
    descriptor_specific_parameters : dict
        Parameters specific to the texture descriptor.

    Returns
    -------
    descriptor_fn : function
        Function that computes the concatenated histogram of the image and its respective keypoints.
    """

    def descriptor_fn(img: NDArray, name_of_the_set: str = "", image_number: int = 0, visualize: bool = False) -> NDArray:
                
        if descriptor_type == "sift":
            keypoints, descriptors = sift_descriptor(
                img,
                descriptor_specific_parameters,
            )
        elif descriptor_type == "orb":
            keypoints, descriptors = orb_descriptor(
                img,
                descriptor_specific_parameters,
            )
        elif descriptor_type == "color_sift":
            keypoints, descriptors = color_sift_descriptor(
                img,
                descriptor_specific_parameters,
            )
            
        else:
            raise ValueError(f"Unsupported texture descriptor: {descriptor_type}")
        
        if visualize:
                visualize_histogram(
                    descriptors.flatten(),
                    name_of_the_set,
                    descriptor_fn.__name__,
                    image_number,
                    channel_labels=[descriptor_type],
                    channel_sizes=[len(descriptors.flatten())]
                )
        return {
            "type": "local",
            "keypoints": keypoints,
            "descriptors": descriptors
        }
        

    descriptor_fn.__name__ = f"{descriptor_type}_{descriptor_specific_parameters}"

    return descriptor_fn


def _kp_to_array(keypoints):
    if keypoints is None or len(keypoints) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(
        [[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in keypoints],
        dtype=np.float32
    )


def sift_descriptor(img: np.ndarray, params):
    if params is None:
        params = {}

    sift = cv2.SIFT_create(
        nfeatures=params.get("nfeatures", 500),
        nOctaveLayers=params.get("nOctaveLayers", 3),
        contrastThreshold=params.get("contrastThreshold", 0.04),
        edgeThreshold=params.get("edgeThreshold", 10),
        sigma=params.get("sigma", 1.6),
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    kp_arr = _kp_to_array(keypoints)
    desc_arr = descriptors.astype(np.float32) if descriptors is not None else np.zeros((0, 128), dtype=np.float32)

    return kp_arr, desc_arr


def orb_descriptor(img: np.ndarray, params):
    if params is None:
        params = {}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    orb = cv2.ORB_create(
        nfeatures=params.get("nfeatures", 500),
        scaleFactor=params.get("scaleFactor", 1.2),
        nlevels=params.get("nlevels", 8),
        edgeThreshold=params.get("edgeThreshold", 31),
        firstLevel=params.get("firstLevel", 0),
        WTA_K=params.get("WTA_K", 2),
        scoreType=params.get("scoreType", cv2.ORB_HARRIS_SCORE),
        patchSize=params.get("patchSize", 31),
        fastThreshold=params.get("fastThreshold", 20),
    )

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    kp_arr = _kp_to_array(keypoints)
    desc_arr = descriptors.astype(np.uint8) if descriptors is not None else np.zeros((0, 32), dtype=np.uint8)

    return kp_arr, desc_arr


def color_sift_descriptor(img: np.ndarray, params):
    if params is None:
        params = {}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    sift = cv2.SIFT_create(
        nfeatures=params.get("nfeatures", 500),
        nOctaveLayers=params.get("nOctaveLayers", 3),
        contrastThreshold=params.get("contrastThreshold", 0.04),
        edgeThreshold=params.get("edgeThreshold", 10),
        sigma=params.get("sigma", 1.6),
    )

    keypoints = sift.detect(v, None)

    if not keypoints:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 384), dtype=np.float32)

    _, desc_h = sift.compute(h, keypoints)
    _, desc_s = sift.compute(s, keypoints)
    _, desc_v = sift.compute(v, keypoints)

    descriptors = np.concatenate((desc_h, desc_s, desc_v), axis=1).astype(np.float32)
    kp_arr = _kp_to_array(keypoints)

    return kp_arr, descriptors