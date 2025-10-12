import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from config import background_removal_config, io_config


BASE_MASK_DIR = io_config.MASKS_DIR / "predicted_masks"
ENSEMBLE_DIR = BASE_MASK_DIR / "ensemble_masks"
PLOTS_DIR = BASE_MASK_DIR / "plots"

BASE_MASK_DIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
if background_removal_config.SAVE_PLOTS:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"left": "cyan", "right": "red", "top": "lime", "bottom": "magenta"}


def first_transition_idx(line, side="left", edge_bootstrap=10, run_window=7,
                         delta_int=20.0, consistency=7):
    """
    Detects the first strong and consistent intensity transition along a 1D signal.
    This defines where the frame border starts from each scanning direction.
    """
    n = len(line)
    if n <= edge_bootstrap + run_window + consistency:
        return None
    if side in ("left", "top"):
        base = np.mean(line[:edge_bootstrap])
        step_range = range(edge_bootstrap, n - run_window - consistency)
        direction = +1
    else:
        base = np.mean(line[-edge_bootstrap:])
        step_range = range(n - edge_bootstrap - run_window - consistency, edge_bootstrap, -1)
        direction = -1

    cumsum = np.cumsum(np.insert(line, 0, 0))  # for O(1) window mean
    win_mean = lambda i: (cumsum[i + run_window] - cumsum[i]) / run_window
    for i in step_range:
        seq = [win_mean(i + j * direction) for j in range(consistency)]
        if all(abs(m - base) > delta_int for m in seq):
            return i
    return None


def detect_edge_points(gray, direction="left", step=2, **params):
    """
    Scans horizontal or vertical lines to detect the first border transition points.
    Returns a list of (x, y) coordinates for each detected edge position.
    """
    H, W = gray.shape
    pts = []
    if direction in ("left", "right"):
        for y in range(0, H, step):
            idx = first_transition_idx(gray[y, :], side=direction, **params)
            if idx is not None:
                pts.append((idx, y))
    else:
        for x in range(0, W, step):
            idx = first_transition_idx(gray[:, x], side=direction, **params)
            if idx is not None:
                pts.append((x, idx))
    return np.array(pts, dtype=float) if len(pts) else np.empty((0, 2))


def filter_outliers(points, axis=0, tol=20):
    """
    Removes points that deviate too far (±tol) from the median value along the chosen axis.
    Helps remove noisy edge detections caused by lighting or texture variation.
    """
    if points.size == 0:
        return points
    med = np.median(points[:, axis])
    kept = points[np.abs(points[:, axis] - med) < tol]
    return kept if kept.size else points


def get_adaptive_channel(img_rgb):
    """
    Chooses between using the Saturation (S) channel or the Luminance (Y)
    channel depending on the variance of saturation.
    If the scene has low color variation (low var(S)), luminance is used instead.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    s = hsv[:, :, 1].astype(np.float32)
    Y = ycbcr[:, :, 0].astype(np.float32)
    var_s = np.var(s)
    if var_s < background_removal_config.VAR_THRESHOLD:
        return Y.astype(np.uint8), "luminance", var_s
    else:
        return s.astype(np.uint8), "saturation", var_s


def fit_linear_line(points, side):
    """
    Fits a linear regression model (y = m*x + b) for each side of the frame,
    converts it to the normalized line equation Ax + By + C = 0.
    """
    if points.size < 2:
        return None
    if side in ("left", "right"):
        X = points[:, 1].reshape(-1, 1)
        y = points[:, 0]
    else:
        X = points[:, 0].reshape(-1, 1)
        y = points[:, 1]
    lr = LinearRegression().fit(X, y)
    m, b = float(lr.coef_[0]), float(lr.intercept_)
    if side in ("left", "right"):
        A, B, C = 1.0, -m, -b
    else:
        A, B, C = -m, 1.0, -b
    norm = np.hypot(A, B)
    return (A / norm, B / norm, C / norm)


def intersect_ABCs(L1, L2):
    """Computes the intersection between two lines given as (A,B,C) coefficients."""
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


def process_image(img_rgb, delta_int):
    """
    Main processing pipeline for a single image and intensity threshold (ΔI).
    Detects edges per side, fits regression lines, computes intersections,
    and generates a binary mask covering the detected frame region.
    """
    gray, mode, var_s = get_adaptive_channel(img_rgb)

    param_grid = list(product(
        background_removal_config.EDGE_BOOTSTRAPS,
        background_removal_config.RUN_WINDOWS,
        [delta_int],
        background_removal_config.CONSISTENCIES,
        background_removal_config.TOLS
    ))

    # Collect edge points for each side
    pts_side = {k: [] for k in ["left", "right", "top", "bottom"]}
    for (edge_bootstrap, run_window, delta_int, consistency, tol) in param_grid:
        for side in pts_side.keys():
            pts = detect_edge_points(
                gray, side, step=background_removal_config.SCAN_STEP,
                edge_bootstrap=edge_bootstrap,
                run_window=run_window,
                delta_int=delta_int,
                consistency=consistency
            )
            if pts.size:
                axis = 0 if side in ("left", "right") else 1
                pts = filter_outliers(pts, axis=axis, tol=tol)
                if pts.size:
                    pts_side[side].append(pts)

    pts_side = {k: np.vstack(v) if len(v) else np.empty((0, 2)) for k, v in pts_side.items()}

    # Fit regression lines and find their intersections
    lines = {side: fit_linear_line(pts_side[side], side) for side in ["left", "right", "top", "bottom"]}
    tl = intersect_ABCs(lines["left"], lines["top"])
    tr = intersect_ABCs(lines["right"], lines["top"])
    br = intersect_ABCs(lines["right"], lines["bottom"])
    bl = intersect_ABCs(lines["left"], lines["bottom"])
    corners = [tl, tr, br, bl]

    # Build binary mask
    H, W, _ = img_rgb.shape
    mask = np.zeros((H, W), np.uint8)
    if all(c is not None for c in corners):
        poly = np.array(corners, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)

    return mask, gray, mode, var_s, pts_side, lines, corners


def run_image(img):
    """
    Runs the full frame edge detection pipeline on a single image.
    It processes the image for all ΔI values and returns the ensemble mask
    built via majority voting across all thresholds.
    """
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        all_masks = []

        # Process one mask per ΔI
        for delta_int in background_removal_config.DELTA_INTS:
            mask, _, _, _, _, _, _ = process_image(img_rgb, delta_int)
            all_masks.append(mask)

        # Majority voting (ensemble)
        stack = np.stack(all_masks, axis=0)
        votes = np.sum(stack > 127, axis=0)
        ensemble_mask = np.where(votes >= len(all_masks) / 2, 255, 0).astype(np.uint8)

        return ensemble_mask

    except Exception as e:
        print(f"Error processing image: {e}")
        return None
