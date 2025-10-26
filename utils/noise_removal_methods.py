import cv2
import numpy as np
import pywt


# NOISE DETECTION ALGORITHM
def estimate_noise_wavelet(img, wavelet='db2', level=1, threshold=5.0):
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    sigma_per_level = []
    for i in range(1, len(coeffs)): 
        LH, HL, HH = coeffs[i]
        sigma = np.median(np.abs(HH)) / 0.6745 
        sigma_per_level.append(sigma)
    
    sigma_est = np.mean(sigma_per_level)
    has_noise = sigma_est > threshold
    return has_noise, sigma_est


# NOISE REMOVAL ALGORTIHM
def denoise_median_bilateral(img_bgr, k_y=3, k_cbcr=5, bilateral_d=5, sigma_color=40, sigma_space=10):
    img_ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img_ycc)

    # Median filters to all the channels withh different hyp
    Y_med = cv2.medianBlur(Y, k_y)
    Cr_med = cv2.medianBlur(Cr, k_cbcr)
    Cb_med = cv2.medianBlur(Cb, k_cbcr)

    # Bilateral to Y
    Y_bilat = cv2.bilateralFilter(Y_med, d=bilateral_d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    img_ycc_deno = cv2.merge([Y_bilat, Cr_med, Cb_med])
    return cv2.cvtColor(img_ycc_deno, cv2.COLOR_YCrCb2BGR)



def main_noise_removal(img: np.ndarray):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0].astype(np.float32)
    detected_noise,_ = estimate_noise_wavelet(Y)

    if detected_noise:
        denoised_image = denoise_median_bilateral(img)
        return denoised_image
    return img