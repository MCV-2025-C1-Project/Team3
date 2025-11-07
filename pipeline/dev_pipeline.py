from config import keypoint_descriptors_config
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
    """Compute descriptors for all dev set images (standardized output)."""

    all_descriptors = []

    for i in tqdm(range(NUMBER_IMAGE_DEV), desc="Dev images processed: "):
        image_path = io_config.dev_image_path(i)
        img = cv2.imread(image_path)
        img = main_noise_removal(img)

        if general_config.REMOVE_BACKGROUND:
            log.info(f"Doing background removal for image: {image_path.name}, please be patient")
            img, mask = get_masks(img)

            # Two paintings
            if len(img) == 2:
                left_descs = [f(img[0], NAME_OF_DEV_SET, i, visualize=False) for f in WANTED_DESCRIPTORS]
                right_descs = [f(img[1], NAME_OF_DEV_SET, i, visualize=False) for f in WANTED_DESCRIPTORS]

                all_descriptors.append({
                    "num_parts": 2,
                    "parts": [left_descs, right_descs]
                })
                continue

            # Single painting: unwrap
            img = img[0]

        # Normal case (no background split)
        descs = [f(img, NAME_OF_DEV_SET, i, visualize=False) for f in WANTED_DESCRIPTORS]

        all_descriptors.append({
            "num_parts": 1,
            "parts": [descs]
        })

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



from sklearn.metrics import precision_recall_fscore_support

def run_dev():
    """
    Full development pipeline with F1 evaluation for all thresholds.
    Generates CSV with one row per (descriptor, distance, distance_type, threshold).
    """
    log.info("Running development pipeline (streaming, memory-efficient)...")

    # === Load ground truth ===
    file_path = io_config.DEV_DIR / "gt_corresps.pkl"
    with open(file_path, "rb") as f:
        ground_truth = pickle.load(f)

    NUMBER_IMAGE_DEV = io_config.count_jpgs(io_config.DEV_DIR)
    NAME_OF_DEV_SET = io_config.DEV_NAME

    # === Prepare descriptors and distances ===
    descriptors_names = [f.__name__ for f in DEV_KEYPOINT_DESCRIPTORS]
    orb = "orb" == descriptors_names[0][:3]
    ALL_DISTANCE_ENTRIES = []

    # Global distances (unchanged)
    for d in general_config.WANTED_DISTANCES:
        if isinstance(d, tuple):
            fn = d[0]; name = fn.__name__; is_tuple = True
        else:
            fn = d; name = fn.__name__; is_tuple = False
        ALL_DISTANCE_ENTRIES.append({
            "name": name, "fn": fn, "kind": "global", "is_tuple": is_tuple
        })

    # Local matchers (unchanged)
    try:
        from utils.local_metrics import (
            sift_match_count,
            sift_match_normalized,
            sift_match_geometric
        )
        local_matchers = [
            ("match_count", sift_match_count),
            ("match_normalized", sift_match_normalized),
            ("match_geometric", sift_match_geometric)
        ]
        distance_types = ["L2", "L1", "HAMMING"]
        for name, fn in local_matchers:
            for d_type in distance_types:
                if orb and (d_type != "HAMMING"):
                    continue
                if not orb and (d_type == "HAMMING"):
                    continue
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
    eval_ks = list(general_config.K_VALUES)
    max_k = max(eval_ks)
    eps = 1e-16

    # === Compute dev descriptors (kept in memory) ===
    log.info("Computing descriptors for all dev images (kept in memory)...")
    raw_all_descriptors = compute_development_descriptors(DEV_KEYPOINT_DESCRIPTORS, NAME_OF_DEV_SET, NUMBER_IMAGE_DEV)

    # === Flatten dev descriptors into queries (I2 / ID_A) ===
    # Each query: { "orig_idx": i, "id": "img_XXXXX" or "img_XXXXX_L", "descs": [desc_dicts...], "gt": [...] }
    queries = []
    for i, entry in enumerate(raw_all_descriptors):
        base_id = f"img_{i:05d}"
        gt_item = ground_truth[i]

        if isinstance(entry, dict) and entry.get("num_parts", 1) == 2:
            # entry['parts'] == [left_descs, right_descs]
            left_descs, right_descs = entry["parts"][0], entry["parts"][1]

            # GT mapping: now ground_truth is a flat list (GT1/flat)
            # handle many GT shapes robustly:
            left_gt = [-1]
            right_gt = [-1]

            if isinstance(gt_item, list):
                if len(gt_item) >= 2:
                    # flat list with two entries -> left and right
                    left_gt = gt_item[0] if isinstance(gt_item[0], list) else [gt_item[0]]
                    right_gt = gt_item[1] if isinstance(gt_item[1], list) else [gt_item[1]]
                elif len(gt_item) == 1:
                    # single GT provided -> assign to both halves
                    left_gt = right_gt = gt_item
            elif isinstance(gt_item, int):
                left_gt = right_gt = [gt_item]
            else:
                # fallback: unknown format -> ignore
                left_gt = [-1]
                right_gt = [-1]

            queries.append({
                "orig_idx": i,
                "id": f"{base_id}_L",
                "descs": left_descs,
                "gt": left_gt
            })
            queries.append({
                "orig_idx": i,
                "id": f"{base_id}_R",
                "descs": right_descs,
                "gt": right_gt
            })
        else:
            # Single part: entry['parts'] == [descs]
            if isinstance(entry, dict) and "parts" in entry:
                descs = entry["parts"][0]
            else:
                # fallback if older format: assume entry is a list of descriptors
                descs = entry

            # Normalize GT for single image: ensure a list
            if isinstance(gt_item, list):
                gt_norm = gt_item
            elif isinstance(gt_item, int):
                gt_norm = [gt_item]
            else:
                gt_norm = [-1]

            if len(gt_norm) > 1:
                gt_norm = [gt_norm[0]]

            queries.append({
                "orig_idx": i,
                "id": base_id,
                "descs": descs,
                "gt": gt_norm
            })

    query_count = len(queries)
    log.info(f"Prepared {query_count} queries from {NUMBER_IMAGE_DEV} dev images (including splits).")

    rows = []

    # === Main loop: per descriptor type (desc_idx) we scan DB file for that descriptor ===
    for desc_idx, desc_name in enumerate(descriptors_names):
        safe_name = sanitize_filename(desc_name)
        print(f"DESC NAME: {desc_name}")

        db_file_path = io_config.KEYPOINT_DESC_DIR / f"{safe_name}.h5"
        if not db_file_path.exists():
            log.warning(f"DB file not found for descriptor {desc_name}: {db_file_path}")
            continue

        # For each distance entry, we will compute heaps per query
        with h5py.File(db_file_path, "r") as h5f:
            num_db_entries = int(h5f.attrs.get("num_images", len([k for k in h5f.keys() if str(k).startswith("img_")])))

            for dist_idx, dist_entry in enumerate(ALL_DISTANCE_ENTRIES):
                dist_kind = dist_entry["kind"]
                dist_fn = dist_entry["fn"]
                is_tuple = dist_entry.get("is_tuple", False)
                distance_type = dist_entry.get("distance_type", "L2")

                # Determine whether to skip this distance entry based on descriptor kind:
                # Inspect a sample dev descriptor for this desc_idx to see its type
                sample_desc = None
                # find first query that has this descriptor index available
                for q in queries:
                    descs = q["descs"]
                    if desc_idx < len(descs):
                        sample_desc = descs[desc_idx]
                        break
                if sample_desc is None:
                    log.warning(f"No sample dev descriptor for desc_idx={desc_idx}, skipping {dist_entry['name']}")
                    continue
                sample_kind = sample_desc.get("type", "global") if isinstance(sample_desc, dict) else "global"
                if sample_kind != dist_kind:
                    # mismatch type (e.g., local matcher vs global descriptor) -> skip
                    continue

                # Initialize heap per query (store top max_k scores). Use min-heap, keep highest scores.
                heaps = [[] for _ in range(query_count)]  # each heap stores tuples (score, db_idx)

                # Stream DB and update heaps
                for db_idx in tqdm(range(num_db_entries), desc=f"{desc_name}/{dist_entry['name']}", leave=False):
                    gname = f"img_{db_idx:05d}"
                    if gname not in h5f:
                        continue
                    grp = h5f[gname]

                    # Build db_entry dict same format as dev entries
                    if "keypoints" in grp and "descriptors" in grp:
                        db_entry = {"type": "local",
                                    "keypoints": np.asarray(grp["keypoints"], dtype=np.float32),
                                    "descriptors": np.asarray(grp["descriptors"])}
                    elif "descriptors" in grp:
                        db_entry = {"type": "global",
                                    "descriptors": np.asarray(grp["descriptors"], dtype=np.float32)}
                    else:
                        continue

                    # For each query compute score and update heap
                    for q_idx, q in enumerate(queries):
                        descs = q["descs"]
                        # if this descriptor index is missing for this query (shouldn't happen) skip
                        if desc_idx >= len(descs):
                            continue
                        dev_entry = descs[desc_idx]
                        # type guard
                        if dev_entry.get("type", "global") != dist_kind:
                            continue

                        # Compute raw score:
                        # - For global distances: dist_fn returns distance (lower better) -> invert sign or use negative
                        # - For tuple-based distance (similarity), follow is_tuple semantics
                        # - For local matchers: dist_fn returns similarity (higher better)
                        if dist_kind == "global":
                            # keep same behavior as original: use negative distance so that larger is better
                            try:
                                raw_val = dist_fn(dev_entry["descriptors"], db_entry["descriptors"])
                            except Exception:
                                raw_val = float("inf")
                            score = -raw_val if not is_tuple else (1.0 / (raw_val + eps))
                        else:
                            # local matcher: may expect additional argument distance_type
                            try:
                                raw_sim = dist_fn(dev_entry, db_entry, distance_type=distance_type)
                            except TypeError:
                                # fallback if matcher ignores distance_type
                                raw_sim = dist_fn(dev_entry, db_entry)
                            except Exception as e:
                                print("Error")
                                raw_sim = 0.0
                            # convert similarity (higher better) into score directly (we keep higher better)
                            score = float(raw_sim)

                        h = heaps[q_idx]
                        if len(h) < max_k:
                            heapq.heappush(h, (score, db_idx))
                        else:
                            # h is min-heap by score; if new score greater than smallest, replace
                            if score > h[0][0]:
                                heapq.heapreplace(h, (score, db_idx))


                # Also prepare per-query sorted entries for AP@K
                per_query_sorted_indices = [None] * query_count
                for unknown_detection_type in keypoint_descriptors_config.DISCARDING_TYPES:
                    for t in unknown_detection_type["thresholds"]:
                        match_scores = []
                        labels = []
                        labels_gt = []
                    
                        for q_idx, q in enumerate(queries):
                            gt = q["gt"]
                            if gt != [-1]:
                                labels_gt.append(1)
                            else:   
                                labels_gt.append(0)
                            h = heaps[q_idx]

                            predicted_unknown = False
                            if len(h) == 0:
                                predicted_unknown = True
                                best_score = 0.0
                                sorted_indices = []
                            else:
                                sorted_entries = sorted(h, key=lambda x: -x[0])
                                best_score = sorted_entries[0][0]

                                if unknown_detection_type["type"] == "threshold":
                                    if best_score < t:
                                        
                                        predicted_unknown = True
                                
                                if unknown_detection_type["type"] == "first_second_ratio":
                                    if len(sorted_entries) > 1:
                                        ratio = best_score / (sorted_entries[1][0] + 1e-8)
                                        if ratio < t:
                                            predicted_unknown = True

                            if predicted_unknown:
                                match_scores.append(-np.inf)
                                labels.append(0)
                                per_query_sorted_indices[q_idx] = [-1]
                                continue
                            else:
                                if h:
                                    # sorted in descending score order
                                    sorted_entries = sorted(h, key=lambda x: -x[0])
                                    best_score = sorted_entries[0][0]
                                    sorted_indices = [e[1] for e in sorted_entries]

                            match_scores.append(best_score)
                            labels.append(1)
                            per_query_sorted_indices[q_idx] = sorted_indices



                        # Convert to numpy arrays for threshold evaluation
                        if len(match_scores) > 0:
                            match_scores_arr = np.array(match_scores)
                            labels_arr = np.array(labels)
                            labels_gt_arr = np.array(labels_gt)
                        else:
                            match_scores_arr = np.array([])
                            labels_arr = np.array([])
                            labels_gt_arr = np.array([])

                        
                        if match_scores_arr.size == 0:
                            f1 = 0.0
                        else:
                            _, _, f1, _ = precision_recall_fscore_support(labels_arr, labels_gt_arr, average='binary', zero_division=0)

                        # Compute mAPs across queries (AP@K)
                        descriptor_scores_sums = np.zeros(len(eval_ks))
                        count_valid = 0
                        for q_idx, q in enumerate(queries):
                            gt = q["gt"]

                            sorted_indices = per_query_sorted_indices[q_idx]
                            # ensure list
                            if sorted_indices is None:
                                sorted_indices = []

                            
                            for k_idx, eval_k in enumerate(eval_ks):
                                preds_k = sorted_indices[:eval_k]
                                score_ap = global_metrics.average_precision_k(gt, preds_k, eval_k)
                                descriptor_scores_sums[k_idx] += score_ap
                            count_valid += 1

                        if count_valid > 0:
                            avg_scores = descriptor_scores_sums / count_valid
                            rows.append({
                                "descriptor": desc_name,
                                "distance": dist_entry["name"],
                                "distance_type": dist_entry.get("distance_type", "N/A"),
                                "discarding_type": unknown_detection_type["type"],
                                "threshold": float(t),
                                "mAP@1": avg_scores[0],
                                "mAP@5": avg_scores[1] if len(avg_scores) > 1 else avg_scores[0],
                                "mean_mAP": float(np.mean(avg_scores)),
                                "F1": float(f1)
                            })
                # end thresholds loop

            # end distance entries loop

        # end with h5f

    # Save CSV
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["descriptor", "distance", "distance_type", "discarding_type", "threshold"])
    io_config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(io_config.RESULTS_DIR / "dev_scores_all_thresholds.csv", index=False)
    log.info(f"✅ Results saved to {io_config.RESULTS_DIR}/dev_scores_all_thresholds.csv (with F1 for all thresholds)")
    log.info("✅ Development pipeline completed successfully with F1 evaluation.")
