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
import h5py

from pipeline.descriptor_creator import sanitize_filename
from utils import global_metrics
from config import io_config, general_config
from config.color_descriptors_config import DEV_COLOR_DESCRIPTORS, DEV_COLOR_DESCRIPTOR_NAMES
from config.texture_descriptors_config import DEV_TEXTURE_DESCRIPTORS, DEV_TEXTURE_DESCRIPTOR_NAMES
from config.keypoint_descriptors_config import DEV_KEYPOINT_DESCRIPTORS, DEV_KEYPOINT_DESCRIPTOR_NAMES
from utils.common import load_precomputed_descriptors
from utils.noise_removal_methods import main_noise_removal
from background_removal.main_background_removal import get_masks
from utils.noise_removal_methods import main_noise_removal



log = logging.getLogger(__name__)


def compute_development_descriptors(WANTED_DESCRIPTORS, NAME_OF_DEV_SET, NUMBER_IMAGE_DEV):
    """Compute descriptors for all dev set images."""
    all_descriptors = []
    for i in tqdm(range(NUMBER_IMAGE_DEV), desc="Dev images processed: "):
        image_path = io_config.dev_image_path(i)
        img = cv2.imread(image_path)
        img = main_noise_removal(img)
        if general_config.REMOVE_BACKGROUND:
            log.info(f"Doing background removal for image: {image_path.name}, please be patient")
            img, mask = get_masks(img)
            
            if len(img) == 2:
                left_descriptors = [f(img[0], NAME_OF_DEV_SET, i, visualize=False) for f in WANTED_DESCRIPTORS]
                right_descriptors = [f(img[1], NAME_OF_DEV_SET, i, visualize=False) for f in WANTED_DESCRIPTORS]
                conjuct = [left_descriptors, right_descriptors]
                all_descriptors.append(conjuct)
                continue
            img = img[0]

        image_descriptors = [f(img, NAME_OF_DEV_SET, i, visualize=False) for f in WANTED_DESCRIPTORS]
        all_descriptors.append(image_descriptors)
    return all_descriptors


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
                    score = global_metrics.average_precision_k(ground_truth[image_num], predictions, eval_k)
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
    3. Compute distances (global + local matchers with multiple distance types).
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

    # Prepare descriptor names
    descriptors_names = [f.__name__ for f in DEV_KEYPOINT_DESCRIPTORS]

    # Build unified distance / matcher list
    ALL_DISTANCE_ENTRIES = []

    # Global distances
    for d in general_config.WANTED_DISTANCES:
        if isinstance(d, tuple):
            fn = d[0]
            name = fn.__name__
            is_tuple = True
        else:
            fn = d
            name = fn.__name__
            is_tuple = False
        ALL_DISTANCE_ENTRIES.append({
            "name": name,
            "fn": fn,
            "kind": "global",
            "is_tuple": is_tuple
        })

    # Local matchers (with multiple distance types)
    try:
        from utils.local_metrics import (
            sift_match_basic,
            sift_match_count,
            sift_match_normalized,
            sift_match_geometric
        )

        local_matchers = [
            ("match_basic", sift_match_basic),
            ("match_count", sift_match_count),
            ("match_normalized", sift_match_normalized),
            ("match_geometric", sift_match_geometric)
        ]

        distance_types = ["L2", "L1", "HAMMING"]

        for name, fn in local_matchers:
            for d_type in distance_types:
                ALL_DISTANCE_ENTRIES.append({
                    "name": f"{name}_{d_type}",
                    "fn": fn,
                    "kind": "local",
                    "is_tuple": False,
                    "distance_type": d_type
                })
    except Exception as e:
        log.warning(f"Local matchers import failed: {e}")

    distances_names = [entry["name"] for entry in ALL_DISTANCE_ENTRIES]


    # Prepare result files and variables
    store = io_config.STORE_RESULTS_TXT_BY_DESCRIPTOR
    result_files = None
    if store:
        io_config.RESULTS_DIR.mkdir(exist_ok=True)
        result_files = [(io_config.RESULTS_DIR / f"{sanitize_filename(name)}_res.txt").open("w") for name in descriptors_names]

    eval_ks = list(general_config.K_VALUES)
    max_k = max(eval_ks)
    descriptor_scores_sums = np.zeros((len(descriptors_names), len(distances_names), len(eval_ks)), dtype=float)
    eps = 1e-16

    # Compute dev descriptors once
    log.info("Computing descriptors for all dev images (kept in memory)...")
    all_descriptors = compute_development_descriptors(DEV_KEYPOINT_DESCRIPTORS, NAME_OF_DEV_SET, NUMBER_IMAGE_DEV)

    # Main loop: descriptor × distance entry
    for desc_idx, desc_name in enumerate(descriptors_names):
        log.info(f"Processing descriptor {desc_idx+1}/{len(descriptors_names)}: {desc_name}")
        db_file_path = io_config.KEYPOINT_DESC_DIR / f"{desc_name}.h5"

        with h5py.File(db_file_path, "r") as h5f:
            num_db_entries = h5f.attrs.get("num_images", len([k for k in h5f.keys() if str(k).startswith("img_")]))

            for dist_idx, dist_entry in enumerate(ALL_DISTANCE_ENTRIES):
                dist_kind = dist_entry["kind"]
                dist_fn = dist_entry["fn"]
                is_tuple = dist_entry.get("is_tuple", False)
                distance_type = dist_entry.get("distance_type", "L2")

                # Detect descriptor kind
                sample_desc = None
                for sample_img_descs in all_descriptors:
                    if isinstance(sample_img_descs, list) and len(sample_img_descs) == 2 and isinstance(sample_img_descs[0], list):
                        sample_desc = sample_img_descs[0][desc_idx]
                        break
                    else:
                        sample_desc = sample_img_descs[desc_idx]
                        break

                if sample_desc is None:
                    log.warning("No sample dev descriptor found — skipping dist entry.")
                    continue

                sample_kind = sample_desc.get("type", "global") if isinstance(sample_desc, dict) else "global"
                if dist_kind != sample_kind:
                    continue

                # Initialize heaps per dev image
                heaps = [[] for _ in range(NUMBER_IMAGE_DEV)]
                for dev_idx in range(NUMBER_IMAGE_DEV):
                    images_descriptors = all_descriptors[dev_idx]
                    if isinstance(images_descriptors, list) and len(images_descriptors) == 2:
                        heaps[dev_idx] = [[], []]

                # Scan DB entries
                for db_idx in tqdm(range(num_db_entries), desc=f"Scanning DB for {desc_name} / {dist_entry['name']}", unit="entries", leave=False):
                    gname = f"img_{db_idx:05d}"
                    if gname not in h5f:
                        continue
                    grp = h5f[gname]

                    if "keypoints" in grp and "descriptors" in grp:
                        db_entry = {"type": "local", "keypoints": np.asarray(grp["keypoints"], np.float32),
                                    "descriptors": np.asarray(grp["descriptors"], np.float32)}
                    elif "descriptors" in grp:
                        db_entry = {"type": "global", "descriptors": np.asarray(grp["descriptors"], np.float32)}
                    else:
                        continue

                    # Compute similarity/distance per dev image
                    for dev_idx in range(NUMBER_IMAGE_DEV):
                        images_descriptors = all_descriptors[dev_idx]

                        if isinstance(images_descriptors, list) and len(images_descriptors) == 2:
                            for sub_i, image_descriptors in enumerate(images_descriptors):
                                dev_entry = image_descriptors[desc_idx]
                                if dev_entry.get("type", "global") != dist_kind:
                                    continue

                                if dist_kind == "global":
                                    raw_score = dist_fn(dev_entry["descriptors"], db_entry["descriptors"])
                                    score = raw_score if not is_tuple else 1.0 / (raw_score + eps)
                                    score = -score 
                                else:
                                    raw_sim = dist_fn(dev_entry, db_entry, distance_type=distance_type)
                                    score = raw_sim  

                                h = heaps[dev_idx][sub_i]
                                if len(h) < max_k:
                                    heapq.heappush(h, (score, db_idx))
                                else:
                                    if score > h[0][0]:
                                        heapq.heapreplace(h, (score, db_idx))

                        else:
                            dev_entry = images_descriptors[desc_idx]
                            if dev_entry.get("type", "global") != dist_kind:
                                continue

                            if dist_kind == "global":
                                raw_score = dist_fn(dev_entry["descriptors"], db_entry["descriptors"])
                                score = raw_score if not is_tuple else 1.0 / (raw_score + eps)
                                score = -score  # menor = mejor
                            else:
                                raw_sim = dist_fn(dev_entry, db_entry, distance_type=distance_type)
                                score = raw_sim

                            h = heaps[dev_idx]
                            if len(h) < max_k:
                                heapq.heappush(h, (score, db_idx))
                            else:
                                if score > h[0][0]:
                                    heapq.heapreplace(h, (score, db_idx))

                # Evaluate AP@K
                for dev_idx in range(NUMBER_IMAGE_DEV):
                    h = heaps[dev_idx]

                    if isinstance(h[0], list):
                        for idx_sub, heap in enumerate(h):
                            sorted_entries = sorted(heap, key=lambda x: -x[0])
                            top_indices = [entry[1] for entry in sorted_entries]
                            for k_idx, eval_k in enumerate(eval_ks):
                                preds_k = top_indices[:eval_k]
                                score = global_metrics.average_precision_k([ground_truth[dev_idx][idx_sub]], preds_k, eval_k)
                                descriptor_scores_sums[desc_idx, dist_idx, k_idx] += 0.5 * score
                        continue

                    sorted_entries = sorted(h, key=lambda x: -x[0])
                    top_indices = [entry[1] for entry in sorted_entries]

                    # Write textual results
                    if store:
                        f_out = result_files[desc_idx]
                        f_out.write(f"Image: {dev_idx:05d}.jpg\n")
                        f_out.write(f"Ground truth: {ground_truth[dev_idx]}\n\n")
                        distance_name = distances_names[dist_idx]
                        f_out.write(f"With distance: {distance_name}\n")
                        f_out.write(f"Top {max_k} images:\n")
                        np.savetxt(f_out, np.array(top_indices)[None], fmt="%d")
                        f_out.write("--------------------------------------------------------------\n")

                    for k_idx, eval_k in enumerate(eval_ks):
                        preds_k = top_indices[:eval_k]
                        score = global_metrics.average_precision_k(ground_truth[dev_idx], preds_k, eval_k)
                        descriptor_scores_sums[desc_idx, dist_idx, k_idx] += score

                if store:
                    f_out = result_files[desc_idx]
                    f_out.write("==============================================================\n")

    if result_files:
        for f in result_files:
            f.close()


    rows = []
    for k_idx, eval_k in enumerate(eval_ks):
        avg_scores = descriptor_scores_sums[:, :, k_idx] / float(NUMBER_IMAGE_DEV)
        for i_desc in range(len(descriptors_names)):
            for j_dist in range(len(distances_names)):
                rows.append({
                    "descriptor": descriptors_names[i_desc],
                    "distance": distances_names[j_dist],
                    f"mAP@{eval_k}": avg_scores[i_desc, j_dist]
                })

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

    log.info("✅ Development pipeline completed successfully (with global + local distances).")
