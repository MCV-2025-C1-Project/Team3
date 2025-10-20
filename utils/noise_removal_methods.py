import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
import bm3d
from config import noise_removal_config

def gaussian_filter(img, ksize, sigma):
    return cv2.GaussianBlur(img, ksize, sigma)


def median_filter(img, ksize):
    return cv2.medianBlur(img, ksize)


def bilateral_filter(img, d, sigmaColor, sigmaSpace):
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)


def nl_means_filter(channel, h, patch_size, patch_distance, fast_mode=True):
    sigma_est = np.mean(estimate_sigma(channel, channel_axis=None))
    denoised = denoise_nl_means(
        channel,
        h=h * sigma_est,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=fast_mode,
        channel_axis=None
    )
    return (denoised * 255).astype(np.uint8)


def bm3d_filter(channel, sigma_psd):
    channel = channel.astype(np.float32) / 255.0
    denoised = bm3d.bm3d(channel, sigma_psd)
    return (denoised * 255).astype(np.uint8)




def main_noise_removal(img: np.ndarray, method_to_detect_noise: str, denoising_method: str):

    if method_to_detect_noise == "Laplacian_Var":
        gf
    

    elif method_to_detect_noise == :
    

    if detected_noise:
        params = noise_removal_config[denoising_method]

        if denoising_method == "Gaussian":
            result = gaussian_filter(img, **params)

        elif denoising_method == "Median":
            result = median_filter(img, **params)

        elif denoising_method == "Bilateral":
            result = bilateral_filter(img, **params)

        elif denoising_method == "non_local_means":
            result = nl_means_filter(img, **params)

        elif denoising_method == "bm3d":
            result = bm3d_filter(img, **params)
        return result
    return img