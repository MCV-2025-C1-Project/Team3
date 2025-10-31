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
                divisions=descriptor_specific_parameters.get("block_size", 8),
                top_k=descriptor_specific_parameters.get("top_k", 20)
            )
        elif descriptor_type == "DWT":
            hist = dwt_descriptor(
                img,
                wavelet=descriptor_specific_parameters.get("wavelet", "db1"),
                levels=descriptor_specific_parameters.get("levels", 2),
                divisions=descriptor_specific_parameters.get("divisions", 8),
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
        

    descriptor_fn.__name__ = f"{descriptor_type}_{descriptor_specific_parameters}"

    return descriptor_fn



def lbp_descriptor(image, P=8, R=1, method='uniform'):
    """
    Compute LBP (Local Binary Pattern) descriptor for an image.
    Returns a fixed-length histogram descriptor.
    """
    img = np.asarray(image)

    # Convert to grayscale if needed
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ensure uint8 dtype
    if not np.issubdtype(img.dtype, np.integer):
        img = (255 * (img / np.max(img))).astype(np.uint8) if img.max() > 1 else img.astype(np.uint8)

    lbp = local_binary_pattern(img, P, R, method=method)

    # Fixed number of bins based on LBP method
    if method == 'uniform':
        n_bins = P + 2
    elif method == 'nri_uniform':
        n_bins = P * (P - 1) + 3
    elif method == 'default' or method == 'ror':
        n_bins = 2 ** P
    else:
        # Fallback: use max code + 1 (only if method is unusual)
        n_bins = int(lbp.max() + 1)

    # Build histogram with fixed number of bins
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), density=False)
    hist = hist.astype(np.float32)

    # Normalize
    s = hist.sum()
    if s > 0:
        hist = hist / s

    # Ensure fixed length (safety padding)
    if hist.shape[0] != n_bins:
        hist = np.pad(hist, (0, n_bins - hist.shape[0]))

    return hist


def zigzag_indices(h, w):
    indices = []
    for s in range(h + w - 1):
        diagonal = []
        for i in range(s + 1):
            j = s - i
            if i < h and j < w:
                diagonal.append((i, j))
        
        # For even diagonals (s=0,2,4...), go down-right (reverse)
        # For odd diagonals (s=1,3,5...), go up-left (normal order)
        if s % 2 == 0:
            diagonal = diagonal[::-1]
        
        indices.extend(diagonal)
    
    return np.array(indices)


def dct_descriptor(image, divisions=8, top_k=20):
    """
    Compute DCT descriptor for an image.

    The descriptor is based on the low-frequency DCT coefficients, which capture
    the main texture and intensity variations of the image.

    Parameters
    ----------
    image : ndarray
        2D grayscale image or 3D color image.
    divisions : int
        The number of blocks in which the image is divided
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

    img = cv2.resize(img, [256, 256])

    img = img.astype(np.float32)
    
    h, w = img.shape
    
    block_h = h // divisions
    block_w = w // divisions
        
    # Crop to full blocks to avoid shape mismatch
    h_crop = (h // block_h) * block_h
    w_crop = (w // block_w) * block_w
    h_start = (h - h_crop) // 2
    w_start = (w - w_crop) // 2
    img = img[h_start:h_start+h_crop, w_start:w_start+w_crop]
    
    blocks = view_as_blocks(img, block_shape=(block_h, block_w))
    blocks = blocks.reshape(-1, block_h, block_w)
    dct_blocks = dctn(blocks, norm='ortho', axes=(-2, -1))
    zz = zigzag_indices(block_h, block_w)[:top_k]
    flat = np.abs(dct_blocks.reshape(blocks.shape[0], -1))# Crop to full blocks to avoid shape mismatch
    desc = np.abs(dct_blocks[:, zz[:, 0], zz[:, 1]]).ravel()

    # Normalize descriptor
    desc /= np.sum(desc) + 1e-8

    return desc



def dwt_descriptor(image, wavelet='db1', levels=2, divisions=8, top_k=20):
    """
    Compute DWT (Discrete Wavelet Transform) descriptor for an image.

    The descriptor is based on multi-level 2D wavelet decomposition, which captures
    texture and intensity variations at multiple scales and orientations.

    Parameters
    ----------
    image : ndarray
        2D grayscale image or 3D color image.
    wavelet : str
        Type of wavelet to use (e.g., 'db1', 'haar', 'sym2', 'coif1').
    levels : int
        Number of decomposition levels for DWT.
    divisions : int
        Number of image blocks to divide the image into before computing DWT.
    top_k : int
        Number of largest coefficients (by absolute value) to keep per block.

    Returns
    -------
    desc : ndarray
        1D normalized DWT-based texture descriptor.
    """

    # Convert to numpy array
    img = np.asarray(image)

    # Convert to grayscale if needed
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize for consistency
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32)

    h, w = img.shape

    # Divide the image into non-overlapping blocks
    block_h = h // divisions
    block_w = w // divisions

    # Crop to full blocks to avoid mismatch
    h_crop = (h // block_h) * block_h
    w_crop = (w // block_w) * block_w
    h_start = (h - h_crop) // 2
    w_start = (w - w_crop) // 2
    img = img[h_start:h_start + h_crop, w_start:w_start + w_crop]

    # Split image into blocks
    blocks = view_as_blocks(img, block_shape=(block_h, block_w))
    blocks = blocks.reshape(-1, block_h, block_w)

    desc_list = []

    # Process each block
    for block in blocks:
        # Perform multi-level 2D wavelet decomposition
        coeffs = pywt.wavedec2(block, wavelet=wavelet, level=levels)

        # Flatten all sub-bands into one array of coefficients
        all_coeffs = []
        for c in coeffs:
            if isinstance(c, tuple):
                # (LH, HL, HH)
                for subband in c:
                    all_coeffs.append(np.abs(subband).ravel())
            else:
                # LL approximation
                all_coeffs.append(np.abs(c).ravel())

        all_coeffs = np.concatenate(all_coeffs)

        # Take top-k largest coefficients (energy-based)
        if len(all_coeffs) > top_k:
            top_values = np.partition(all_coeffs, -top_k)[-top_k:]
        else:
            top_values = all_coeffs

        # Sort top coefficients for consistency
        top_values = np.sort(top_values)[::-1]

        # Normalize block descriptor
        top_values /= np.sum(top_values) + 1e-8

        desc_list.append(top_values)

    # Concatenate all block descriptors
    desc = np.concatenate(desc_list)

    # Global normalization
    desc /= np.sum(desc) + 1e-8

    return desc

