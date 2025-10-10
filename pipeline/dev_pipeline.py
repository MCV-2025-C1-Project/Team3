import cv2
import pandas as pd
import numpy as np
import pickle
import logging
import heapq
import math
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import metrics
from config import io_config, general_config
from config.color_descriptors_config import DEV_COLOR_DESCRIPTORS, DEV_COLOR_DESCRIPTOR_NAMES
from utils.common import load_precomputed_descriptors


log = logging.getLogger(__name__)


def compute_development_descriptors(WANTED_DESCRIPTORS, NAME_OF_DEV_SET, NUMBER_IMAGE_DEV):
    """Compute descriptors for all dev set images."""
    all_descriptors = []
    for i in tqdm(range(NUMBER_IMAGE_DEV), desc="Dev images processed: "):
        image_path = io_config.dev_image_path(i)
        img = cv2.imread(image_path)
        image_descriptors = [f(img, NAME_OF_DEV_SET, i, visualize=False) for f in WANTED_DESCRIPTORS]
        all_descriptors.append(image_descriptors)
    return all_descriptors

def compute_distances(all_descriptors, precomputed_descriptors, WANTED_DISTANCES):
    """Compute distances between dev and BBDD descriptors."""
    all_metrics = []
    for objective_image in all_descriptors:
        image_metrics = []
        for idx, descriptor in enumerate(objective_image):
            found_metrics = []
            objective_descriptors = precomputed_descriptors[idx]
            for distance_function in WANTED_DISTANCES:
                if isinstance(distance_function, tuple):
                    distance_function = distance_function[0]
                distances = [distance_function(descriptor, obj) for obj in objective_descriptors]
                found_metrics.append(distances)
            image_metrics.append(found_metrics)
        all_metrics.append(image_metrics)
    return all_metrics


def write_results(all_metrics, ground_truth, descriptors_names, distances_names, K=5, store = io_config.STORE_RESULTS_TXT_BY_DESCRIPTOR):
    """Save top-K retrieval results for each descriptor and distance."""
    if store:
        io_config.RESULTS_DIR.mkdir(exist_ok=True)
        result_files = [(io_config.RESULTS_DIR / f"{name}_res.txt").open("w") for name in descriptors_names]

        for image_num, image_metrics in enumerate(all_metrics):
            for descriptor_type, metric in enumerate(image_metrics):
                f = result_files[descriptor_type]
                f.write(f"Image: {image_num:05d}.jpg\n")
                gt_images = ground_truth[image_num]
                f.write(f"Ground truth: {gt_images}\n\n")
                for distance_type, distance_name in enumerate(distances_names):
                    f.write(f"With distance: {distance_name}\n")
                    f.write(f"Top {K} images:\n")
                    distances = np.array(metric[distance_type])
                    if isinstance(general_config.WANTED_DISTANCES[distance_type], tuple):
                        distances = 1 / (distances + 1e-16)
                    top_k_res = np.argsort(distances)[:K]
                    np.savetxt(f, top_k_res[None], fmt="%d")
                    f.write("--------------------------------------------------------------\n")
                f.write("==============================================================\n")
        for f in result_files:
            f.close()


def resume_results(all_metrics, ground_truth, descriptors_names, distances_names,
                   wanted_distances, NUMBER_IMAGE_DEV, eval_ks):
    """
    Visual summary: AP@K heatmaps for descriptors vs distances.
    Also saves a CSV with numerical results sorted by average mAP.
    """
    Path("results").mkdir(exist_ok=True)
    rows = [] 

    for eval_k in eval_ks:
        descriptor_scores = [[0 for _ in range(len(distances_names))] for _ in range(len(descriptors_names))]
        
        for image_num, image_metrics in enumerate(all_metrics):
            for descriptor_type, metric in enumerate(image_metrics):
                for distance_type, distance_name in enumerate(distances_names):
                    distances = np.array(metric[distance_type])
                    if isinstance(wanted_distances[distance_type], tuple):
                        distances = 1 / (distances + 1e-16)
                    predictions = np.argsort(distances)[:eval_k]
                    score = metrics.average_precision_k(ground_truth[image_num], predictions, eval_k)
                    descriptor_scores[descriptor_type][distance_type] += score

        # Media por descriptor-distancia
        for i in range(len(descriptor_scores)):
            for j in range(len(descriptor_scores[0])):
                descriptor_scores[i][j] /= NUMBER_IMAGE_DEV
                rows.append({
                    "descriptor": descriptors_names[i],
                    "distance": distances_names[j],
                    f"mAP@{eval_k}": descriptor_scores[i][j]
                })

        # Save heatmaps
        fig, ax = plt.subplots()
        ax.set_title(f"Scores with K={eval_k}", fontsize=12, fontweight="bold")
        cax = ax.matshow(descriptor_scores, cmap="viridis")
        ax.set_xticks(range(len(distances_names)))
        ax.set_yticks(range(len(descriptors_names)))
        ax.set_xticklabels([n.replace("_", " ") for n in distances_names], rotation=90)
        ax.set_yticklabels(descriptors_names)
        for i in range(len(descriptors_names)):
            for j in range(len(descriptor_scores[0])):
                ax.text(j, i, f"{descriptor_scores[i][j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        plt.colorbar(cax)
        plt.savefig(io_config.RESULTS_DIR / f"obtained_scores_k{eval_k}.png", dpi=300, bbox_inches="tight")

    df = pd.DataFrame(rows)
    df = df.groupby(["descriptor", "distance"]).first().reset_index()

    map_cols = [c for c in df.columns if c.startswith("mAP@")]
    df["mean_mAP"] = df[map_cols].mean(axis=1)

    #   Sort by the mean of both columns
    df = df.sort_values(by="mean_mAP", ascending=False)

    df.to_csv(io_config.RESULTS_DIR / "dev_scores.csv", index=False)
    log.info(f"✅ Results saved to {io_config.RESULTS_DIR}/dev_scores.csv (sorted by mean mAP)")




def run_dev():
    """
    Run the complete development pipeline:
    1. Compute descriptors for dev images.
    2. Load precomputed DB descriptors.
    3. Compute distances.
    4. Write results.
    5. Render visual summary.
    """
    log.info("Running development pipeline (streaming, memory-efficient)...")

    # Load ground truth
    file_path = io_config.DEV_DIR / "gt_corresps.pkl"
    with open(file_path, "rb") as f:
        ground_truth = pickle.load(f)

    # Setup basic info
    NUMBER_IMAGE_DEV = io_config.count_jpgs(io_config.DEV_DIR)
    NAME_OF_DEV_SET = io_config.DEV_NAME

    # Prepare names and files
    descriptors_names = [f.__name__ for f in DEV_COLOR_DESCRIPTORS]
    distances_names = [
        d[0].__name__ if isinstance(d, tuple) else d.__name__
        for d in general_config.WANTED_DISTANCES
    ]

    # Prepare result files (one per descriptor) if configured
    store = io_config.STORE_RESULTS_TXT_BY_DESCRIPTOR
    result_files = None
    if store:
        io_config.RESULTS_DIR.mkdir(exist_ok=True)
        result_files = [(io_config.RESULTS_DIR / f"{name}_res.txt").open("w") for name in descriptors_names]

    # We'll compute top-K where K is the maximum of K_VALUES to satisfy all eval Ks
    eval_ks = list(general_config.K_VALUES)
    max_k = max(eval_ks)

    # Prepare accumulators for descriptor x distance x K (sum of AP@K across dev images)
    descriptor_scores_sums = np.zeros((len(descriptors_names), len(distances_names), len(eval_ks)), dtype=float)

    eps = 1e-16

    def stream_top_k_indices(descriptor_vec, db_file_path, distance_fn, is_tuple, K):
        """Stream DB descriptors from file and keep K smallest scores (following previous code's inversion for tuple metrics). Returns indices sorted by score asc."""
        heap = []  # will store (-score, db_index) to maintain max-heap of smallest scores
        with open(db_file_path, "r") as fh:
            for db_idx, line in enumerate(fh):
                try:
                    db_vec = np.fromstring(line, sep=" ")
                except Exception:
                    continue
                raw_score = distance_fn(db_vec, descriptor_vec) if not is_tuple else distance_fn(db_vec, descriptor_vec)
                # In original code, when distance was a tuple they later inverted by 1/(distances+eps)
                score = raw_score
                if is_tuple:
                    # raw_score here is a similarity; make it comparable with distances by inversion
                    score = 1.0 / (raw_score + eps)

                if len(heap) < K:
                    heapq.heappush(heap, (-score, db_idx))
                else:
                    current_max = -heap[0][0]
                    if score < current_max:
                        heapq.heapreplace(heap, (-score, db_idx))

        # Extract indices sorted by increasing score
        sorted_entries = sorted(heap, key=lambda x: -x[0])
        top_indices = [entry[1] for entry in sorted_entries]
        return top_indices

    # Compute descriptors for all dev images once (these are small compared to DB)
    log.info("Computing descriptors for all dev images (kept in memory)...")
    all_descriptors = compute_development_descriptors(DEV_COLOR_DESCRIPTORS, NAME_OF_DEV_SET, NUMBER_IMAGE_DEV)

    # For each descriptor type and distance, scan the DB file only once and update top-K heaps for every dev image
    for desc_idx, desc_name in enumerate(descriptors_names):
        log.info(f"Processing descriptor {desc_idx+1}/{len(descriptors_names)}: {desc_name}")
        db_file_path = io_config.COLOR_DESC_DIR / f"{desc_name}.txt"

        # For each distance metric we will create per-dev heaps
        for dist_idx, dist_entry in enumerate(general_config.WANTED_DISTANCES):
            dist_fn = dist_entry[0] if isinstance(dist_entry, tuple) else dist_entry
            is_tuple = isinstance(dist_entry, tuple)

            # Initialize a heap per dev image (store up to max_k best matches)
            heaps = [[] for _ in range(NUMBER_IMAGE_DEV)]  # each is a max-heap via (-score, db_idx)

            # Stream DB once
            with open(db_file_path, "r") as fh:
                for db_idx, line in enumerate(tqdm(fh, desc=f"Scanning DB for {desc_name} / {dist_idx}", unit="lines", leave=False)):
                    try:
                        db_vec = np.fromstring(line, sep=" ")
                    except Exception:
                        continue

                    # For each dev image compute distance and update its heap
                    for dev_idx in range(NUMBER_IMAGE_DEV):
                        dev_vec = all_descriptors[dev_idx][desc_idx]
                        raw_score = dist_fn(dev_vec, db_vec)
                        score = raw_score if not is_tuple else 1.0 / (raw_score + eps)

                        h = heaps[dev_idx]
                        if len(h) < max_k:
                            heapq.heappush(h, (-score, db_idx))
                        else:
                            current_max = -h[0][0]
                            if score < current_max:
                                heapq.heapreplace(h, (-score, db_idx))

            # After scanning DB, extract top-K for each dev image and compute AP@k
            for dev_idx in range(NUMBER_IMAGE_DEV):
                h = heaps[dev_idx]
                sorted_entries = sorted(h, key=lambda x: -x[0])  # increasing score
                top_indices = [entry[1] for entry in sorted_entries]

                # optionally write textual results
                if store:
                    f_out = result_files[desc_idx]
                    f_out.write(f"Image: {dev_idx:05d}.jpg\n")
                    f_out.write(f"Ground truth: {ground_truth[dev_idx]}\n\n")
                    distance_name = distances_names[dist_idx]
                    f_out.write(f"With distance: {distance_name}\n")
                    f_out.write(f"Top {max_k} images:\n")
                    np.savetxt(f_out, np.array(top_indices)[None], fmt="%d")
                    f_out.write("--------------------------------------------------------------\n")

                # accumulate AP@k scores
                for k_idx, eval_k in enumerate(eval_ks):
                    preds_k = top_indices[:eval_k]
                    score = metrics.average_precision_k(ground_truth[dev_idx], preds_k, eval_k)
                    descriptor_scores_sums[desc_idx, dist_idx, k_idx] += score

            # close block for this distance
            if store:
                # mark separator between distances per descriptor
                f_out = result_files[desc_idx]
                f_out.write("==============================================================\n")

    # Close result files
    if result_files:
        for f in result_files:
            f.close()

    # Build rows and save CSV and heatmaps similar to previous resume_results
    rows = []
    for k_idx, eval_k in enumerate(eval_ks):
        # average by number of dev images
        avg_scores = descriptor_scores_sums[:, :, k_idx] / float(NUMBER_IMAGE_DEV)
        for i_desc in range(len(descriptors_names)):
            for j_dist in range(len(distances_names)):
                rows.append({
                    "descriptor": descriptors_names[i_desc],
                    "distance": distances_names[j_dist],
                    f"mAP@{eval_k}": avg_scores[i_desc, j_dist]
                })

        # save heatmap
        fig, ax = plt.subplots()
        ax.set_title(f"Scores with K={eval_k}", fontsize=12, fontweight="bold")
        cax = ax.matshow(avg_scores, cmap="viridis")
        ax.set_xticks(range(len(distances_names)))
        ax.set_yticks(range(len(descriptors_names)))
        ax.set_xticklabels([n.replace("_", " ") for n in distances_names], rotation=90)
        ax.set_yticklabels(descriptors_names)
        for ii in range(len(descriptors_names)):
            for jj in range(len(distances_names)):
                ax.text(jj, ii, f"{avg_scores[ii, jj]:.3f}", ha="center", va="center", color="white", fontsize=8)
        plt.colorbar(cax)
        plt.savefig(io_config.RESULTS_DIR / f"obtained_scores_k{eval_k}.png", dpi=300, bbox_inches="tight")

    df = pd.DataFrame(rows)
    df = df.groupby(["descriptor", "distance"]).first().reset_index()
    map_cols = [c for c in df.columns if c.startswith("mAP@")]
    df["mean_mAP"] = df[map_cols].mean(axis=1)
    df = df.sort_values(by="mean_mAP", ascending=False)
    df.to_csv(io_config.RESULTS_DIR / "dev_scores.csv", index=False)
    log.info(f"✅ Results saved to {io_config.RESULTS_DIR}/dev_scores.csv (sorted by mean mAP)")

    log.info("Development pipeline completed successfully (streaming).")
