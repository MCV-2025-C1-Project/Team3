import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from config import background_removal_config, io_config
from scipy.ndimage import gaussian_filter

BASE_MASK_DIR = io_config.MASKS_DIR / "predicted_masks"
ENSEMBLE_DIR = BASE_MASK_DIR / "ensemble_masks"
PLOTS_DIR = BASE_MASK_DIR / "plots"

BASE_MASK_DIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
if background_removal_config.SAVE_PLOTS:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"left": "cyan", "right": "red", "top": "lime", "bottom": "magenta"}



def detect_first_white(line, side="left"):
    """Detects the first white pixel (non-zero) in a 1D binary line."""
    n = len(line)
    if side in ("left", "top"):
        for i in range(n):
            if line[i] > 0:
                return i
    else:  # right or bottom
        for i in range(n - 1, -1, -1):
            if line[i] > 0:
                return i
    return None


def detect_edge_points(binary, side="left", step=2):
    """Scans the binary gradient image horizontally or vertically and returns (x, y) coordinates."""
    H, W = binary.shape
    pts = []
    if side in ("left", "right"):
        for y in range(0, H, step):
            idx = detect_first_white(binary[y, :], side)
            if idx is not None:
                pts.append((idx, y))
    else:
        for x in range(0, W, step):
            idx = detect_first_white(binary[:, x], side)
            if idx is not None:
                pts.append((x, idx))
    return np.array(pts, float) if len(pts) else np.empty((0, 2))


def filter_outliers_priority_border(points, side, tol=20):
    """Median-based outlier filtering."""
    if points.size == 0:
        return points

    axis = 0 if side in ("left", "right") else 1
    coord = points[:, axis]

    if side == "left":
        dist = coord
    elif side == "right":
        dist = points[:, 0].max() - coord
    elif side == "top":
        dist = points[:, 1]
    elif side == "bottom":
        dist = points[:, 1].max() - coord

    med = np.median(dist)
    mask = np.abs(dist - med) < tol
    filtered = points[mask]

    if filtered.size < 10:
        order = np.argsort(np.abs(dist - med))
        filtered = points[order[:5]]

    return filtered


def fit_linear_line(points, side):
    """Fits a linear regression line (y = m*x + b) and returns normalized (A,B,C)."""
    if points.size < 2:
        return None
    if side in ("left", "right"):
        X = points[:, 1].reshape(-1, 1)
        y = points[:, 0]
    else:
        X = points[:, 0].reshape(-1, 1)
        y = points[:, 1]
    lr = LinearRegression().fit(X, y)
    m, b = lr.coef_[0], lr.intercept_
    if side in ("left", "right"):
        A, B, C = 1.0, -m, -b
    else:
        A, B, C = -m, 1.0, -b
    norm = np.hypot(A, B)
    return (A / norm, B / norm, C / norm)


def intersect_ABCs(L1, L2):
    """Computes the intersection point between two lines Ax + By + C = 0."""
    if L1 is None or L2 is None:
        return None
    A1, B1, C1 = L1
    A2, B2, C2 = L2
    M = np.array([[A1, B1], [A2, B2]], dtype=float)
    rhs = -np.array([C1, C2], dtype=float)
    if abs(np.linalg.det(M)) < 1e-8:
        return None
    x, y = np.linalg.solve(M, rhs)
    return int(round(x)), int(round(y))



def retinex_shadow_removal(img_rgb, blur_kernel=51):
    """
    Applies Single-Scale Retinex (SSR) for shadow and illumination removal.
    - Estimates illumination by Gaussian blur.
    - Divides the original image by the illumination estimate.
    """
    img_float = img_rgb.astype(np.float32) / 255.0
    illumination = gaussian_filter(img_float, sigma=blur_kernel/6)
    illumination = np.clip(illumination, 1e-3, 1.0)
    corrected = img_float / illumination
    corrected = np.clip(corrected / np.max(corrected), 0, 1)
    return (corrected * 255).astype(np.uint8)



def process_image(grad_mag, threshold):
    """
    Processes a single RGB image for a given gradient threshold.
    Steps:
      1. Apply Retinex for shadow removal.
      2. Compute Sobel gradient on Cr channel (YCrCb space).
      3. Binarize and clean.
      4. Detect edge points per side.
      5. Fit lines and compute intersections.
      6. Build mask polygon.
    """
    

    _, binary = cv2.threshold(grad_mag, threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    H, W = binary.shape
    sides = ["left", "right", "top", "bottom"]

    pts_side = {s: detect_edge_points(binary, s, background_removal_config.SCAN_STEP) for s in sides}
    for s in sides:
        pts_side[s] = filter_outliers_priority_border(pts_side[s], s)

    lines = {side: fit_linear_line(pts_side[side], side) for side in sides}
    tl = intersect_ABCs(lines["left"], lines["top"])
    tr = intersect_ABCs(lines["right"], lines["top"])
    br = intersect_ABCs(lines["right"], lines["bottom"])
    bl = intersect_ABCs(lines["left"], lines["bottom"])
    corners = [tl, tr, br, bl]

    mask = np.zeros((H, W), np.uint8)
    if all(c is not None for c in corners):
        cv2.fillPoly(mask, [np.array(corners, np.int32)], 255)

    return mask, grad_mag, binary, corners


def run_image_v2(img):
    """
    Runs the full frame edge detection pipeline on a single image,
    using Retinex for illumination correction and ensemble fusion
    based on interpolated averaging with median area thresholding.
    """
    try:
        
        #Putted out of the function to compute only once
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        all_masks = []
        img_rgb = retinex_shadow_removal(img_rgb)

        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(img_hsv)

        grad_x = cv2.Sobel(s, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(s, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_mag = cv2.convertScaleAbs(grad_mag)

        for threshold in background_removal_config.G_THRESHOLDS:
            mask, grad, binary, corners = process_image(grad_mag, threshold)
            all_masks.append(mask)

            if background_removal_config.SAVE_PLOTS:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(grad, cmap="gray")
                axs[0].set_title(f"|∇Cr| — th={threshold}")
                axs[0].axis("off")

                axs[1].imshow(binary, cmap="gray")
                axs[1].set_title("Binarized Gradient")
                axs[1].axis("off")

                axs[2].imshow(mask, cmap="gray")
                axs[2].set_title("Generated Mask")
                axs[2].axis("off")

                plt.tight_layout()
                plot_path = PLOTS_DIR / f"debug_grad_{threshold}.png"
                plt.savefig(plot_path, dpi=150)
                plt.close(fig)

        # "Median mask ensemble"
        stack = np.stack(all_masks, axis=0).astype(np.float32)
        avg_mask = np.mean(stack, axis=0)

        # Compute total white area (number of white pixels) per mask
        areas = [np.sum(m > 127) for m in all_masks]
        target_area = int(np.median(areas))  # choose the median area

        # Determine the adaptive threshold in the averaged mask
        flat = np.sort(avg_mask.ravel())
        if target_area >= len(flat):
            threshold_val = flat[0] 
        else:
            threshold_val = flat[-target_area]
        print("Hi Aleix")

        # Binarize averaged mask with adaptive threshold
        ensemble_mask = np.where(avg_mask >= threshold_val, 255, 0).astype(np.uint8)

        return ensemble_mask

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

