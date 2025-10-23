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
from scipy.fftpack import dct


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



def generic_texture_descriptor(descriptor_type: str,
                       descriptor_specific_parameters: dict,
                       denoising_method: list[str] = ["none"],
                       denoising_kernel_size: list[int] = [3]):
    """
    Factory function that creates a descriptor function from a given configuration.

    Parameters
    ----------
    color_space : str
        Texture descriptor typw (e.g., "LBP").
    descriptor_specific_parameters : dict
        Parameters specific to the texture descriptor.
    denoising_method : list[str]
        Denoising method to apply before computing the descriptor. Options: "none", "gaussian", "median", "bilateral".
    denoising_kernel_size : list[int]
        Kernel size for the denoising method.

    Returns
    -------
    descriptor_fn : function
        Function that computes the concatenated histogram of the image.
    """

    def descriptor_fn(img: NDArray, name_of_the_set: str = "", image_number: int = 0, visualize: bool = False) -> NDArray:
        
        # Apply denoising if specified
        if denoising_method == "gaussian":
            converted = cv2.GaussianBlur(img, (denoising_kernel_size[0], denoising_kernel_size[0]), 0)
        elif denoising_method == "median":
            converted = cv2.medianBlur(img, denoising_kernel_size[0])
        elif denoising_method == "bilateral":
            converted = cv2.bilateralFilter(img, d=denoising_kernel_size[0], sigmaColor=75, sigmaSpace=75)
        else:
            converted = img.copy()

        if general_config.REMOVE_BACKGROUND:
            converted, _ = main_background_removal(converted)
        
        if descriptor_type == "LBP":
            hist = lbp_descriptor(
                img,
                P=descriptor_specific_parameters.get("P", 8),
                R=descriptor_specific_parameters.get("R", 1),
                method=descriptor_specific_parameters.get("method", "uniform")
            )
        elif descriptor_type == "DCT":
            hist = dct_descriptor(
                img,
                block_size=descriptor_specific_parameters.get("block_size", 8),
                top_k=descriptor_specific_parameters.get("top_k", 20)
            )
            
        else:
            raise ValueError(f"Unsupported texture descriptor: {descriptor_type}")
        
        if visualize:
                visualize_histogram(
                    hist,
                    name_of_the_set,
                    descriptor_fn.__name__,
                    image_number,
                    channel_labels=[descriptor_type],
                    channel_sizes=[len(hist)]
                )
        return hist
        

    descriptor_fn.__name__ = f"{descriptor_type}_{descriptor_specific_parameters}_denoise-{denoising_method}{'-'.join(map(str, denoising_kernel_size))}"

    return descriptor_fn



def lbp_descriptor(image, P=8, R=1, method='uniform'):
    """
    Compute LBP (Local Binary Pattern) descriptor for an image.

    Parameters
    ----------
    image : ndarray
            2D grayscale image or 3D color image (H, W) or (H, W, C)
    P :     int
            Number of circularly symmetric neighbor set points (default=8).
    R :     int
            Radius of circle (in pixels) (default=1).
    method: Method for LBP. 
            Options: 'default', 'ror', 'uniform', 'var' (default='uniform').

    Returns
    -------
    hist : ndarray
        Normalized histogram of LBP codes.
    """

    # convert to numpy array
    img = np.asarray(image)

    # Convert to grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ensure integer dtype (uint8)
    if not np.issubdtype(img.dtype, np.integer):
        img = (255 * (img / np.max(img))).astype(np.uint8) if img.max() > 1 else img.astype(np.uint8)


    lbp = local_binary_pattern(img, P, R, method=method)

    # decide number of histogram bins according to method as recommended by skimage docs
    if method == 'uniform':
        n_bins = P + 2
    else:
        n_bins = int(lbp.max() + 1) 

    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), density=False)
    hist = hist.astype(np.float32)
    # normalize histogram
    s = hist.sum()
    if s > 0:
        hist = hist / s
    return hist


def dct_descriptor(image, block_size=8, top_k=20):
    """
    Compute DCT descriptor for an image.

    The descriptor is based on the low-frequency DCT coefficients, which capture
    the main texture and intensity variations of the image.

    Parameters
    ----------
    image : ndarray
        2D grayscale image or 3D color image.
    block_size : int
        Size of the block for block-wise DCT.
        If None, computes the DCT for the whole image.
    top_k : int
        Number of lowest-frequency DCT coefficients to keep.
        These represent the most relevant texture information.

    Returns
    -------
    desc : ndarray
        1D normalized DCT-based texture descriptor.
    """

    # Convert to numpy array
    img = np.asarray(image)

    # Convert to grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)

    # block-wise DCT
    if block_size is not None:
        h, w = img.shape
        desc_list = []
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = img[i:i + block_size, j:j + block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                # Flatten and sort by frequency importance (top-left corner = low frequencies)
                dct_flat = dct_block.flatten()
                # Sort by zigzag order
                idx = np.argsort(np.abs(dct_flat))[-top_k:]
                desc_list.append(dct_flat[idx])
        desc = np.concatenate(desc_list)
    else:
        # global DCT
        dct_img = dct(dct(img.T, norm='ortho').T, norm='ortho')
        dct_flat = dct_img.flatten()
        idx = np.argsort(np.abs(dct_flat))[-top_k:]
        desc = dct_flat[idx]

    # Normalize descriptor
    desc = np.abs(desc)
    s = np.sum(desc)
    if s > 0:
        desc = desc / s

    return desc