import cv2
import numpy as np
import pickle
from config import general_config, io_config
from config.keypoint_descriptors_config import PREDICTING_KEYPOINT_DESCRIPTORS
from utils.common import load_precomputed_descriptors
from utils.noise_removal_methods import main_noise_removal
from background_removal.main_background_removal import get_masks
import logging
import h5py
from tqdm import tqdm
from pipeline.descriptor_creator import sanitize_filename

log = logging.getLogger(__name__)

def predict_and_save_results():
    log.info("🚀 Running test pipeline for keypoint descriptors...")

    images = sorted([p for p in io_config.TEST_DIR.iterdir() if p.suffix.lower() == ".jpg"])
    log.info(f"Found {len(images)} test images in {io_config.TEST_DIR}")

    descriptor_func, matcher_func = PREDICTING_KEYPOINT_DESCRIPTORS[0]
    safe_name = sanitize_filename(descriptor_func.__name__)

    db_path = io_config.KEYPOINT_DESC_DIR / f"{safe_name}.h5"
    if not db_path.exists():
        raise FileNotFoundError(f"❌ Descriptor file not found: {db_path}")

    with h5py.File(db_path, "r") as f:
        precomputed_descriptors = load_precomputed_descriptors([f])

    for method_idx, (used_descriptor, used_matcher) in enumerate(PREDICTING_KEYPOINT_DESCRIPTORS, start=1):
        desc_name = sanitize_filename(used_descriptor.__name__)
        matcher_name = used_matcher.__name__
        log.info(f"🔹 Method {method_idx}: {desc_name} + {matcher_name}")

        output_dir = io_config.RESULTS_DIR / io_config.TEST_NAME / f"method{method_idx}"
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for image in tqdm(images, desc=f"Processing {desc_name}"):
            img = cv2.imread(str(image))
            img = main_noise_removal(img)

            if general_config.REMOVE_BACKGROUND:
                img, mask = get_masks(img)
                img_list = img if len(img) == 2 else [img[0]]
            else:
                img_list = [img]

            total_res = []

            for painting in img_list:
                query_desc = used_descriptor(painting)
                match_scores = []

                for db_desc in precomputed_descriptors[method_idx - 1]:
                    try:
                        score = used_matcher(query_desc, db_desc, distance_type="L1")
                        threshold = 10
                    except TypeError:
                        score = used_matcher(query_desc, db_desc)
                        threshold = None
                    except Exception as e:
                        log.warning(f"Error comparing descriptors: {e}")
                        score = 0.0
                    match_scores.append(score)

                match_scores = np.array(match_scores)

                if threshold is not None:
                    best_score = float(np.max(match_scores))
                    if best_score < threshold:
                        total_res.append([-1])
                        continue
                    top_k_idx = np.argsort(-match_scores)[:general_config.TOP_K_TEST]
                else:
                    top_k_idx = np.argsort(match_scores)[:general_config.TOP_K_TEST]

                total_res.append(top_k_idx.tolist())

            # ✅ FORMATO FINAL CORRECTO
            if all(res == [-1] for res in total_res):
                results.append([[-1]])                          # unknown
            elif len(total_res) == 1:
                results.append([total_res[0]])                  # una pintura
            elif len(total_res) == 2:
                results.append(total_res)                       # dos pinturas
            else:
                raise ValueError(f"Unexpected structure in {image.name}: {total_res}")

        output_path = output_dir / "result.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        log.info(f"✅ Saved results to {output_path} ({len(results)} images)")

    log.info("🏁 All predictions completed successfully.")
