import numpy as np
import cv2
import matplotlib.pyplot as plt


class Descriptors:
    """
    Collection of descriptors of an image
    """

    ## Discuss usage of numpy
    @staticmethod
    def bgr_histogram(image_path: str, save_img: bool = False) -> list:
        """
        Input:
        -image_path (str): Path to the image
        -save_img (bool): Save the histogram as an image (default: False)


        Returns:
        -hist: BGR histogram concatenated by colors
        """

        img = cv2.imread(image_path)
        img = img.astype(int)

        hist = [0] * (256 * 3)

        for row in img:
            for col in row:
                hist[col[0]] += 1
                hist[col[1] + 256] += 1
                hist[col[2] + 512] += 1

        hist = np.array(hist, dtype=np.float32)
        hist /= (hist.sum() + 1e-7)


        if save_img:
            plt.figure(figsize=(12, 4))
            plt.bar(range(768), hist, width=1.0, color="black")
            plt.xlabel("Concatenated BGR bins (0-255 B, 256-511 G, 512-767 R)")
            plt.ylabel("Frequency")
            plt.title("Concatenated BGR Histogram")
            plt.savefig("results/bgr_histogram.png", dpi=300, bbox_inches="tight")

        return hist

    @staticmethod
    def grayscale_histogram(image_path: str, save_img: bool = False) -> np.ndarray:
        """
        Input:
        -image_path (str): Path to the image
        -save_img (bool): Save the histogram as an image (default: False)

        Returns:
        -hist: Grayscale histogram
        """

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(int)
        hist = [0] * 256
        for row in img:
            for pixel in row:
                hist[pixel] += 1
        hist = np.array(hist, dtype=np.float32)

        # # or using cv2.calcHist
        # hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        # hist = hist.flatten()

        # Normalisation
        hist /= hist.sum() + 1e-7

        if save_img:
            plt.figure(figsize=(10, 4))
            plt.bar(range(256), hist, width=1.0, color="gray")
            plt.title("Grayscale Histogram")
            plt.xlabel("Grayscale Intensity (0-255)")
            plt.ylabel("Frequency")
            plt.savefig("results/grayscale_histogram.png", dpi=300, bbox_inches="tight")

        return hist


descriptors = Descriptors()
image_path = "data/BBDD/bbdd_00000.jpg"
print(f"bgr histogram: {descriptors.bgr_histogram(image_path, True)}")
print(f"grayscale histogram: {descriptors.grayscale_histogram(image_path, True)}")
