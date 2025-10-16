"""
Collection of descriptors of an image
"""

import numpy as np
from config import io_config
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

    output_dir = io_config.HIST_DIR / name_of_the_set / histogram_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{image_number:05d}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.close()


def compute_histogram(
    img: NDArray[np.uint8],
    bins: int = 256,
    value_range: tuple[int, int] = (0, 256),
    levels: list[int] = [1],
) -> NDArray[np.float32]:
    """
    Compute a histogram for a single-channel image.

    Parameters
    ----------
    - img : NDArray[np.uint8]
        Input single-channel image.
    - bins : int
        Number of bins in the histogram.
    - value_range : tuple of int
        The (min, max) value range for the histogram.
    - levels : list[int]
        List of hierarchical levels to compute.
        Example:
            [1]      -> histogram of the whole image.
            [2]      -> histograms of 4 quadrants (2x2 grid).
            [1, 2, 3]-> histogram of whole image + 4 quadrants + 9 subregions (3x3 grid).

    Returns
    -------
    - hist : NDArray[np.float32]
        Concatenated normalized histograms from all specified levels.
    """

    def _compute_single_hist(image: NDArray[np.uint8]) -> NDArray[np.float32]:
        hist, _ = np.histogram(image, bins=bins, range=value_range)
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        return hist

    H, W = img.shape
    hist_list = []

    for level in levels:
        n_rows = n_cols = level
        row_step = H // n_rows
        col_step = W // n_cols

        for i in range(n_rows):
            for j in range(n_cols):
                r_start, r_end = i * row_step, (i + 1) * row_step if i < n_rows - 1 else H
                c_start, c_end = j * col_step, (j + 1) * col_step if j < n_cols - 1 else W

                region = img[r_start:r_end, c_start:c_end]
                hist_list.append(_compute_single_hist(region))

    return np.concatenate(hist_list, axis=0)



def generic_color_descriptor(color_space: str,
                       channels: list[str],
                       bins: list[int],
                       ranges: list[tuple[int, int]],
                       weights: list[float],
                       hierarchical_levels: list[int],
                       denoising_method: str = "none",
                       denoising_kernel_size: int = 3):
    """
    Factory function that creates a descriptor function from a given configuration.

    Parameters
    ----------
    color_space : str
        Color space (e.g., "rgb", "hsv", "lab", "ycbcr", "gray").
    channels : list[str]
        Channel names to use.
    bins : list[int]
        Number of bins per channel.
    ranges : list[tuple[int,int]]
        Value ranges for each channel.
    weights : list[float]
        Weights to apply to each channel.

    Returns
    -------
    descriptor_fn : function
        Function that computes the concatenated histogram of the image.
    """

    def descriptor_fn(img: NDArray, name_of_the_set: str = "", image_number: int = 0, visualize: bool = False) -> NDArray:
        if color_space == "rgb":
            converted = img
        elif color_space == "gray":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif color_space == "hsv":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == "lab":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        elif color_space == "ycbcr":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        else:
            raise ValueError(f"Unsupported color space: {color_space}")
        
        # Apply denoising if specified
        if denoising_method == "gaussian":
            converted = cv2.GaussianBlur(converted, (denoising_kernel_size, denoising_kernel_size), 0)
        elif denoising_method == "median":
            converted = cv2.medianBlur(converted, denoising_kernel_size)
        elif denoising_method == "bilateral":
            converted = cv2.bilateralFilter(converted, d=denoising_kernel_size, sigmaColor=75, sigmaSpace=75)


        if color_space == "gray":
            channel_imgs = [converted]
        else:
            channel_imgs = cv2.split(converted)

        # Select only the configured channels
        selected_imgs = []
        for ch in channels:
            if ch in ["B", "G", "R"]:
                selected_imgs.append(channel_imgs[["B","G","R"].index(ch)])
            elif ch in ["H", "S", "V"]:
                selected_imgs.append(channel_imgs[["H","S","V"].index(ch)])
            elif ch in ["L", "A", "B"]:
                selected_imgs.append(channel_imgs[["L","A","B"].index(ch)])
            elif ch in ["Y", "Cr", "Cb"]:
                selected_imgs.append(channel_imgs[["Y","Cr","Cb"].index(ch)])
            elif ch == "Gray":
                selected_imgs.append(channel_imgs[0])
            else:
                raise ValueError(f"Unknown channel {ch} for {color_space}")

        hists = []
        for img_ch, b, r, w in zip(selected_imgs, bins, ranges, weights):
            hist = compute_histogram(img_ch, bins=b, value_range=r, levels=hierarchical_levels)
            hists.append(hist * w)

        final_hist = np.concatenate(hists)

        if visualize:
            visualize_histogram(
                final_hist,
                name_of_the_set,
                descriptor_fn.__name__,
                image_number,
                channel_labels=channels,
                channel_sizes=bins
            )


        return final_hist

    descriptor_fn.__name__ = (
    f"{color_space}_{'_'.join(channels)}"
    f"_bins{'-'.join(map(str,bins))}"
    f"_w{'-'.join(map(str,weights))}"
    f"_hier{'-'.join(map(str,hierarchical_levels))}"
    )
    return descriptor_fn


def compute_histogram_2d(
    img_x: NDArray[np.uint8],
    img_y: NDArray[np.uint8],
    bins: tuple[int, int] = (256, 256),
    ranges: tuple[tuple[int, int], tuple[int, int]] = ((0, 256), (0, 256)),
    levels: list[int] = [1],
) -> NDArray[np.float32]:
    """
    Compute concatenated 2D histograms for the specified hierarchical levels.
    Returns a 1D float32 vector (concatenation of flattened normalized 2D histograms).
    """
    def _compute_single_2d_hist(ix: NDArray[np.uint8], iy: NDArray[np.uint8]) -> NDArray[np.float32]:
        hist2d, _, _ = np.histogram2d(ix.ravel(), iy.ravel(), bins=bins, range=ranges)
        hist2d = hist2d.astype(np.float32)
        hist2d /= (hist2d.sum() + 1e-7)
        return hist2d.ravel()

    H, W = img_x.shape
    hist_list = []
    for level in levels:
        n_rows = n_cols = level
        row_step = H // n_rows
        col_step = W // n_cols
        for i in range(n_rows):
            for j in range(n_cols):
                r_start, r_end = i * row_step, (i + 1) * row_step if i < n_rows - 1 else H
                c_start, c_end = j * col_step, (j + 1) * col_step if j < n_cols - 1 else W
                region_x = img_x[r_start:r_end, c_start:c_end]
                region_y = img_y[r_start:r_end, c_start:c_end]
                hist_list.append(_compute_single_2d_hist(region_x, region_y))
    return np.concatenate(hist_list, axis=0)


def generic_color_descriptor_2d(
    color_space: str,
    channels: list[str],
    bins: list[int],
    ranges: list[tuple[int, int]],
    weights: list[float],
    hierarchical_levels: list[int],
):
    """
    Variant of generic_color_descriptor that computes 2D joint histograms.
    Assumes `channels` contains exactly 2 channel names corresponding to the pair.
    - `bins` should be a list/tuple of two ints: [bins_x, bins_y]
    - `ranges` should be a list/tuple of two (min,max) tuples: [range_x, range_y]
    - `weights` may be a single value (applied to the joint 2D histogram)
    """
    def descriptor_fn(img: NDArray, name_of_the_set: str = "", image_number: int = 0, visualize: bool = False) -> NDArray:
        if color_space == "rgb":
            converted = img
        elif color_space == "gray":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif color_space == "hsv":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == "lab":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        elif color_space == "ycbcr":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        else:
            raise ValueError(f"Unsupported color space: {color_space}")

        if color_space == "gray":
            channel_imgs = [converted]
        else:
            channel_imgs = cv2.split(converted)

        if len(channels) != 2:
            raise ValueError("generic_color_descriptor_2d expects exactly 2 channels")

        # map channel names to arrays (reuse same mapping logic)
        selected = []
        for ch in channels:
            if ch in ["B", "G", "R"]:
                selected.append(channel_imgs[["B","G","R"].index(ch)])
            elif ch in ["H", "S", "V"]:
                selected.append(channel_imgs[["H","S","V"].index(ch)])
            elif ch in ["L", "A", "B"]:
                selected.append(channel_imgs[["L","A","B"].index(ch)])
            elif ch in ["Y", "Cr", "Cb"]:
                selected.append(channel_imgs[["Y","Cr","Cb"].index(ch)])
            elif ch == "Gray":
                selected.append(channel_imgs[0])
            else:
                raise ValueError(f"Unknown channel {ch} for {color_space}")

        # bins and ranges as two-element structures for 2D hist
        bins_2d = (bins[0], bins[1])
        ranges_2d = (ranges[0], ranges[1])
        joint_hist = compute_histogram_2d(selected[0], selected[1], bins=bins_2d, ranges=ranges_2d, levels=hierarchical_levels)


        # apply a single weight if provided, otherwise 1.0
        w = float(weights[0]) if len(weights) > 0 else 1.0
        # Flatten for compatibility with 1D functions
        final_hist = joint_hist.ravel().astype(np.float32) * w

        if visualize:
            visualize_histogram(
                final_hist,
                name_of_the_set,
                descriptor_fn.__name__,
                image_number,
                channel_labels=channels,
                channel_sizes=[final_hist.size],
            )

        return final_hist

    descriptor_fn.__name__ = (
        f"{color_space}_{'_'.join(channels)}_2d"
        f"_bins{'-'.join(map(str,bins))}"
        f"_w{'-'.join(map(str,weights))}"
        f"_hier{'-'.join(map(str,hierarchical_levels))}"
    )
    return descriptor_fn


def compute_histogram_3d(
    img_x: NDArray[np.uint8],
    img_y: NDArray[np.uint8],
    img_z: NDArray[np.uint8],
    bins: tuple[int, int, int] = (256, 256, 256),
    ranges: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = ((0, 256), (0, 256), (0, 256)),
    levels: list[int] = [1],
) -> NDArray[np.float32]:
    """
    Compute concatenated 3D joint histograms for the specified hierarchical levels.
    Returns a flattened 1D float32 vector (concatenation of normalized 3D histograms).

    Parameters
    - img_x, img_y, img_z : single-channel images (same shape)
    - bins: tuple of three ints (bins per dim)
    - ranges: tuple of three (min,max) tuples
    - levels: hierarchical levels (like compute_histogram)
    """
    def _compute_single_3d_hist(ix: NDArray[np.uint8], iy: NDArray[np.uint8], iz: NDArray[np.uint8]) -> NDArray[np.float32]:
        # build samples of shape (n_pixels, 3)
        samples = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1).astype(np.float32)
        hist3d, _ = np.histogramdd(samples, bins=bins, range=ranges)
        hist3d = hist3d.astype(np.float32)
        hist3d /= (hist3d.sum() + 1e-7)
        return hist3d.ravel()

    H, W = img_x.shape
    hist_list = []
    for level in levels:
        n_rows = n_cols = level
        row_step = H // n_rows
        col_step = W // n_cols
        for i in range(n_rows):
            for j in range(n_cols):
                r_start, r_end = i * row_step, (i + 1) * row_step if i < n_rows - 1 else H
                c_start, c_end = j * col_step, (j + 1) * col_step if j < n_cols - 1 else W
                region_x = img_x[r_start:r_end, c_start:c_end]
                region_y = img_y[r_start:r_end, c_start:c_end]
                region_z = img_z[r_start:r_end, c_start:c_end]
                hist_list.append(_compute_single_3d_hist(region_x, region_y, region_z))

    if len(hist_list) == 0:
        return np.array([], dtype=np.float32)
    return np.concatenate(hist_list, axis=0)


def generic_color_descriptor_3d(
    color_space: str,
    channels: list[str],
    bins: list[int],
    ranges: list[tuple[int, int]],
    weights: list[float],
    hierarchical_levels: list[int],
):
    """
    Descriptor factory that computes 3D joint histograms across three channels.
    - channels must contain exactly 3 channel names.
    - bins must have length 3, ranges length 3.
    Returns flattened float32 vector compatible with 1D postprocessing.
    """
    def descriptor_fn(img: NDArray, name_of_the_set: str = "", image_number: int = 0, visualize: bool = False) -> NDArray:
        if color_space == "rgb":
            converted = img
        elif color_space == "gray":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif color_space == "hsv":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == "lab":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        elif color_space == "ycbcr":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        else:
            raise ValueError(f"Unsupported color space: {color_space}")

        if color_space == "gray":
            channel_imgs = [converted]
        else:
            channel_imgs = cv2.split(converted)

        if len(channels) != 3:
            raise ValueError("generic_color_descriptor_3d expects exactly 3 channels")

        selected = []
        for ch in channels:
            if ch in ["B", "G", "R"]:
                selected.append(channel_imgs[["B","G","R"].index(ch)])
            elif ch in ["H", "S", "V"]:
                selected.append(channel_imgs[["H","S","V"].index(ch)])
            elif ch in ["L", "A", "B"]:
                selected.append(channel_imgs[["L","A","B"].index(ch)])
            elif ch in ["Y", "Cr", "Cb"]:
                selected.append(channel_imgs[["Y","Cr","Cb"].index(ch)])
            elif ch == "Gray":
                selected.append(channel_imgs[0])
            else:
                raise ValueError(f"Unknown channel {ch} for {color_space}")

        bins_3d = (bins[0], bins[1], bins[2])
        ranges_3d = (ranges[0], ranges[1], ranges[2])
        joint_hist = compute_histogram_3d(selected[0], selected[1], selected[2], bins=bins_3d, ranges=ranges_3d, levels=hierarchical_levels)

        # apply a single scalar weight (if provided) to the whole joint histogram
        w = float(weights[0]) if len(weights) > 0 else 1.0
        final_hist = joint_hist.ravel().astype(np.float32) * w

        if visualize:
            visualize_histogram(
                final_hist,
                name_of_the_set,
                descriptor_fn.__name__,
                image_number,
                channel_labels=channels,
                channel_sizes=[final_hist.size],
            )

        return final_hist

    descriptor_fn.__name__ = (
        f"{color_space}_{'_'.join(channels)}_3d"
        f"_bins{'-'.join(map(str,bins))}"
        f"_w{'-'.join(map(str,weights))}"
        f"_hier{'-'.join(map(str,hierarchical_levels))}"
    )
    return descriptor_fn