import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from pathlib import Path
from config import background_removal_config, io_config



PLOTS_DIR = io_config.MASK_DIR_ALG_X / "plots"
MASK_OUTPUTS = MASK_DIR_ALG_X = io_config.MASKS_DIR/"predicted_masks"

MASK_OUTPUTS.mkdir(parents=True, exist_ok=True)
if background_removal_config.SAVE_PLOTS:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def first_transition_idx(line, side="left", edge_bootstrap=10, run_window=7,
                         delta_int=20.0, consistency=7):
    """Detects the first consistent intensity transition along a 1D profile."""
    n = len(line)
    cumsum = np.cumsum(np.insert(line, 0, 0))  # for O(1) window mean
    win_mean = lambda i: (cumsum[i + run_window] - cumsum[i]) / run_window
    if side in ("left", "top"):
        base = np.mean(line[:edge_bootstrap])
        step_range = range(edge_bootstrap, n - run_window - consistency)
        direction = +1
    else:
        base = np.mean(line[-edge_bootstrap:])
        step_range = range(n - edge_bootstrap - run_window - consistency, edge_bootstrap, -1)
        direction = -1

    for i in step_range:
        seq = [win_mean(i + j * direction) for j in range(consistency)]
        if all(abs(m - base) > delta_int for m in seq):
            return i
    return None


def detect_edge_points(gray, direction="left", step=2, **params):
    """Scans image lines (horizontal/vertical) and collects edge points."""
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
    return np.array(pts, dtype=float)


def filter_outliers(points, axis=0, tol=20):
    """Filters points beyond ±tol from median along chosen axis."""
    if points.size == 0:
        return points
    med = np.median(points[:, axis])
    return points[np.abs(points[:, axis] - med) < tol]


def get_adaptive_channel(img_rgb):
    """Chooses between Saturation (S) or Luminance (Y) depending on S variance."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    s = hsv[:, :, 1].astype(np.float32)
    
    var_s = np.var(s)

    if var_s < background_removal_config.VAR_THRESHOLD:
        ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        Y = ycbcr[:, :, 0].astype(np.float32)
        return Y.astype(np.uint8), "luminance", var_s
    else:
        return s.astype(np.uint8), "saturation", var_s


def process_image(img_rgb, param_grid):
    """Performs edge detection and builds the final mask through majority voting."""
    gray, mode, var_s = get_adaptive_channel(img_rgb)
    H, W = gray.shape
    accum_mask = np.zeros((H, W), np.uint16)
    collected_edges = None

    for (edge_bootstrap, run_window, delta_int, consistency, tol) in param_grid:
        edges_raw = {
            "left": detect_edge_points(gray, "left", step=background_removal_config.SCAN_STEP,
                                       edge_bootstrap=edge_bootstrap, run_window=run_window,
                                       delta_int=delta_int, consistency=consistency),
            "right": detect_edge_points(gray, "right", step=background_removal_config.SCAN_STEP,
                                        edge_bootstrap=edge_bootstrap, run_window=run_window,
                                        delta_int=delta_int, consistency=consistency),
            "top": detect_edge_points(gray, "top", step=background_removal_config.SCAN_STEP,
                                      edge_bootstrap=edge_bootstrap, run_window=run_window,
                                      delta_int=delta_int, consistency=consistency),
            "bottom": detect_edge_points(gray, "bottom", step=background_removal_config.SCAN_STEP,
                                         edge_bootstrap=edge_bootstrap, run_window=run_window,
                                         delta_int=delta_int, consistency=consistency),
        }

        if collected_edges is None:
            collected_edges = edges_raw

        edges = {
            k: filter_outliers(v, axis=(0 if k in ["left", "right"] else 1), tol=tol)
            for k, v in edges_raw.items()
        }

        left_x = int(np.median(edges["left"][:, 0])) if edges["left"].size else 0
        right_x = int(np.median(edges["right"][:, 0])) if edges["right"].size else W - 1
        top_y = int(np.median(edges["top"][:, 1])) if edges["top"].size else 0
        bottom_y = int(np.median(edges["bottom"][:, 1])) if edges["bottom"].size else H - 1

        right_x, bottom_y = max(right_x, left_x + 1), max(bottom_y, top_y + 1)

        mask = np.zeros((H, W), np.uint8)
        mask[top_y:bottom_y, left_x:right_x] = 1
        accum_mask += mask

    threshold_votes = int(len(param_grid) * 0.5)
    final_mask = np.where(accum_mask >= threshold_votes, 255, 0).astype(np.uint8)

    return final_mask, gray, mode, var_s, collected_edges


# MAIN LOOP

param_grid = list(product(background_removal_config.EDGE_BOOTSTRAPS,
                              background_removal_config.RUN_WINDOWS,
                              background_removal_config.DELTA_INTS,
                              background_removal_config.CONSISTENCIES,
                              background_removal_config.TOLS))

def run_image(img):

    
    try:
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        final_mask, gray, mode, var_s, edges = process_image(img_rgb, param_grid)

        return final_mask

    except Exception as e:
        print(f"Error processing {e}")
    
    

def run_dataset(dev_dir, max_imgs=30):
    """Processes all .jpg images in a directory and saves masks & plots."""
    db_images = sorted([p for p in Path(dev_dir).iterdir() if p.suffix.lower() == ".jpg"])
    n_imgs = min(max_imgs, len(db_images))
    param_grid = list(product(background_removal_config.EDGE_BOOTSTRAPS,
                              background_removal_config.RUN_WINDOWS,
                              background_removal_config.DELTA_INTS,
                              background_removal_config.CONSISTENCIES,
                              background_removal_config.TOLS))

    for img_path in tqdm(db_images[:n_imgs], desc="Processing images"):
        try:
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            final_mask, gray, mode, var_s, edges = process_image(img_rgb, param_grid)

            # save masks
            mask_name = MASK_OUTPUTS / f"{img_path.stem}.png"
            cv2.imwrite(str(mask_name), final_mask)

            # plots
            if background_removal_config.SAVE_PLOTS:
                fig, axs = plt.subplots(1, 3, figsize=(16, 5))
                plt.suptitle(f"{img_path.name} | Mode: {mode} | Var(S)={var_s:.1f}",
                             fontsize=11, fontweight="bold")

                # Original + Detected Edges
                axs[0].imshow(img_rgb)
                axs[0].set_title("Original + Detected Edges")
                axs[0].axis("off")

                if edges:
                    color_map = {"left": "cyan", "right": "red", "top": "lime", "bottom": "magenta"}
                    for k, c in color_map.items():
                        if edges[k].size:
                            axs[0].scatter(edges[k][:, 0], edges[k][:, 1],
                                           s=10, c=c, label=k, alpha=0.8)
                    axs[0].legend(loc="lower right", fontsize=8)

                # Channel Used
                axs[1].imshow(gray, cmap="gray")
                axs[1].set_title(f"{mode.capitalize()} channel")
                axs[1].axis("off")

                # Final Mask
                axs[2].imshow(final_mask, cmap="gray")
                axs[2].set_title("Final Mask (Majority Vote)")
                axs[2].axis("off")

                plt.tight_layout()

                plot_name = PLOTS_DIR / f"{img_path.stem}_plot.png"
                plt.savefig(plot_name, dpi=150, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()

            print(f"Saved mask alg X: {mask_name.name}")

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

