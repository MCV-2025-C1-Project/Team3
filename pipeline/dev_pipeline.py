import cv2
import numpy as np
import pickle
import logging
from pathlib import Path
import matplotlib.pyplot as plt

from utils import metrics
from config import io_config, general_config
from config.color_descriptors_config import WANTED_COLOR_DESCRIPTORS, WANTED_COLOR_DESCRIPTORS_NAMES
from utils.common import load_precomputed_descriptors


log = logging.getLogger(__name__)


def compute_development_descriptors(WANTED_DESCRIPTORS, NAME_OF_DEV_SET, NUMBER_IMAGE_DEV):
    """Compute descriptors for all dev set images."""
    all_descriptors = []
    for i in range(NUMBER_IMAGE_DEV):
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


def write_results(all_metrics, ground_truth, descriptors_names, distances_names, K=5):
    """Save top-K retrieval results for each descriptor and distance."""
    io_config.RESULTS_DIR.mkdir(exist_ok=True)
    result_files = [(io_config.RESULTS_DIR / f"{name}_res.txt").open("w") for name in descriptors_names]

    for image_num, image_metrics in enumerate(all_metrics):
        for descriptor_type, metric in enumerate(image_metrics):
            f = result_files[descriptor_type]
            f.write(f"Image: {image_num:05d}.jpg\n")
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
                   WANTED_DISTANCES, NUMBER_IMAGE_DEV, eval_ks):
    """Visual summary: AP@K heatmaps for descriptors vs distances."""
    Path("results").mkdir(exist_ok=True)
    for eval_k in eval_ks:
        descriptor_scores = [[0 for _ in range(len(distances_names))] for _ in range(len(descriptors_names))]
        for image_num, image_metrics in enumerate(all_metrics):
            for descriptor_type, metric in enumerate(image_metrics):
                for distance_type, distance_name in enumerate(distances_names):
                    distances = np.array(metric[distance_type])
                    if isinstance(WANTED_DISTANCES[distance_type], tuple):
                        distances = 1 / (distances + 1e-16)
                    predictions = np.argsort(distances)[:eval_k]
                    score = metrics.average_precision_k(ground_truth[image_num], predictions, eval_k)
                    descriptor_scores[descriptor_type][distance_type] += score
        for i in range(len(descriptor_scores)):
            for j in range(len(descriptor_scores[0])):
                descriptor_scores[i][j] /= NUMBER_IMAGE_DEV

        fig, ax = plt.subplots()
        ax.set_title(f"Scores with K={eval_k}", fontsize=12, fontweight="bold")
        cax = ax.matshow(descriptor_scores, cmap="viridis")
        ax.set_xticks(range(len(distances_names)))
        ax.set_yticks(range(len(descriptors_names)))
        ax.set_xticklabels([n.replace("_", " ") for n in distances_names], rotation=90)
        ax.set_yticklabels(descriptors_names)
        for i in range(len(descriptors_names)):
            for j in range(len(distances_names)):
                ax.text(j, i, f"{descriptor_scores[i][j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        plt.colorbar(cax)
        plt.savefig(io_config.RESULTS_DIR / f"obtained_scores_k{eval_k}.png", dpi=300, bbox_inches="tight")


def run_dev():
    """
    Run the complete development pipeline:
    1. Compute descriptors for dev images.
    2. Load precomputed DB descriptors.
    3. Compute distances.
    4. Write results.
    5. Render visual summary.
    """
    log.info("Running development pipeline...")

    # Load ground truth
    file_path = io_config.DEV_DIR / "gt_corresps.pkl"
    with open(file_path, "rb") as f:
        ground_truth = pickle.load(f)

    # Setup basic info
    NUMBER_IMAGE_DEV = io_config.count_jpgs(io_config.DEV_DIR)
    NAME_OF_DEV_SET = io_config.DEV_NAME

    # Compute descriptors for dev set
    all_descriptors = compute_development_descriptors(
        WANTED_COLOR_DESCRIPTORS, NAME_OF_DEV_SET, NUMBER_IMAGE_DEV
    )

    # Load precomputed DB descriptors
    descriptors_names = [f.__name__ for f in WANTED_COLOR_DESCRIPTORS]
    distances_names = [
        d[0].__name__ if isinstance(d, tuple) else d.__name__
        for d in general_config.WANTED_DISTANCES
    ]
    files = [(io_config.COLOR_DESC_DIR / f"{name}.txt").open("r") for name in descriptors_names]
    precomputed_descriptors = load_precomputed_descriptors(files)

    # Compute distances
    all_metrics = compute_distances(all_descriptors, precomputed_descriptors, general_config.WANTED_DISTANCES)

    # Write textual results
    write_results(all_metrics, ground_truth, descriptors_names, distances_names)

    # Visual summary (AP@K heatmaps)
    resume_results(
        all_metrics, ground_truth, descriptors_names, distances_names,
        general_config.WANTED_DISTANCES, NUMBER_IMAGE_DEV, general_config.K_VALUES
    )

    log.info("Development pipeline completed successfully.")
