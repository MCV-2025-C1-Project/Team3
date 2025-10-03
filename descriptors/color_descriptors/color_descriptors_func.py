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

def generic_color_descriptor(color_space: str,
                       channels: list[str],
                       bins: list[int],
                       ranges: list[tuple[int, int]],
                       weights: list[float]):
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
            hist = compute_histogram(img_ch, bins=b, value_range=r)
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
    )
    return descriptor_fn

def mixed_concat_descriptor(configs: list[dict]):
    """
    Creates a mixed descriptor from multiple color spaces (like concatenate rgb.hsv....).

    Parameters
    ----------
    configs : list[dict]
        Each dict must contain {color_space, channels, bins, ranges, weights}.

    Returns
    -------
    descriptor_fn : function
        Descriptor that concatenates histograms from all provided configs.
    """

    def descriptor_fn(img: NDArray, name_of_the_set: str, image_number: int, visualize: bool = False) -> NDArray:
        hists = []
        for cfg in configs:
            fn = generic_color_descriptor(**cfg)
            h = fn(img, name_of_the_set, image_number, visualize=False)
            hists.append(h)
        final_hist = np.concatenate(hists)

        if visualize:
            visualize_histogram(
                final_hist,
                name_of_the_set,
                descriptor_fn.__name__,
                image_number
            )

        return final_hist

    parts = []
    for cfg in configs:
        part = (
            f"{cfg['color_space']}_{'_'.join(cfg['channels'])}"
            f"_bins{'-'.join(map(str,cfg['bins']))}"
            f"_w{'-'.join(map(str,cfg['weights']))}"
        )
        parts.append(part)

    descriptor_fn.__name__ = "MIXED_" + "__".join(parts)
    return descriptor_fn

def mixed_sum_descriptor(configs: list[tuple]):
    """
    Creates a mixed descriptor from multiple color spaces (like concatenate rgb.hsv....).

    Parameters
    ----------
    configs : list[tuple]
        Each tuple must contain a dict with {color_space, channels, bins, ranges, weights}
        and an int that sets its weight. All the dicts must have the same number of bins.

    Returns
    -------
    descriptor_fn : function
        Descriptor that concatenates histograms from all provided configs.
    """

    def descriptor_fn(img: NDArray, name_of_the_set: str, image_number: int, visualize: bool = False) -> NDArray:
        hist = np.array([0] * (configs[0][0]['bins'][0] * len(configs[0][0]['bins'])), dtype=np.float32)
        for cfg in configs:
            fn = generic_color_descriptor(**(cfg[0]))
            h = fn(img, name_of_the_set, image_number, visualize=False)
            hist = np.add(hist, h*cfg[1])
        final_hist = hist

        if visualize:
            visualize_histogram(
                final_hist,
                name_of_the_set,
                descriptor_fn.__name__,
                image_number
            )

        return final_hist

    parts = []
    for cfg in configs:
        part = (
            f"{cfg[0]['color_space']}_{'_'.join(cfg[0]['channels'])}"
            f"_w-{cfg[1]}"
        )
        parts.append(part)

    descriptor_fn.__name__ = "MIXED_SUM" + "__".join(parts)
    return descriptor_fn