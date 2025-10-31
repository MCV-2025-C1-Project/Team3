import cv2
import numpy as np
import pickle
from config import general_config
from config import io_config
from config.color_descriptors_config import PREDICTING_COLOR_DESCRIPTORS
from config.texture_descriptors_config import PREDICTING_TEXTURE_DESCRIPTORS
from utils.common import load_precomputed_descriptors
from utils.noise_removal_methods import main_noise_removal
from background_removal.main_background_removal import get_masks
import logging
import h5py
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

def predict_and_save_results():
    """
    Compute predictions for all test queries and save results as PKL files.
    One PKL per method in PREDICTING_COLOR_DESCRIPTORS.

    Output files:
    results/method1/result.pkl
    results/method2/result.pkl
    """
    images = sorted([p for p in io_config.TEST_DIR.iterdir() if p.suffix.lower() == ".jpg"])

    descriptors_names = [f[0].__name__ for f in PREDICTING_TEXTURE_DESCRIPTORS]
    files = [h5py.File(io_config.TEXTURE_DESC_DIR / f"{name}.h5", 'r') for name in descriptors_names]
    precomputed_descriptors = load_precomputed_descriptors(files)

    for method_idx, (used_descriptor, used_distance) in enumerate(PREDICTING_TEXTURE_DESCRIPTORS, start=1):
        log.info(f"Running prediction for method {method_idx}: {used_descriptor.__name__} + {used_distance.__name__}")

        output_dir = io_config.RESULTS_DIR / io_config.TEST_NAME / f"masks"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for image in images:
            img = cv2.imread(str(image))
            img = main_noise_removal(img)
            if general_config.REMOVE_BACKGROUND:
                log.info(f"Doing background removal for image: {image.name}, please be patient")
                img, mask = get_masks(img)
                if len(mask) == 2:
                    mask = np.hstack(mask)
                else:
                    mask = mask[0]
                if general_config.SAVE_BACKGROUND_MASK:
                    cv2.imwrite(output_dir / f"{image.with_suffix(".png").name}", mask)
                log.info(f"Done")
                
            
            if len(img) == 2:
                
                total_res = []
                
                for painting in img:
                    query_descriptor = used_descriptor(painting)
                    distances = []
                    for db_descriptor in precomputed_descriptors[method_idx - 1]:
                        d = used_distance(db_descriptor, query_descriptor)
                        if used_distance.__name__ in ["hellinger_kernel"]:
                            d = 1 / (d + 1e-16)
                        distances.append(d)
                        
                    distances = np.array(distances)
                    top_k_idx = np.argsort(distances)[:general_config.TOP_K_TEST]
                    total_res.append(top_k_idx.tolist())
                    
                results.append(total_res)
                continue
            
            if len(img) < 2:
                img = img[0]
                
            query_descriptor = used_descriptor(img)

            distances = []
            for db_descriptor in precomputed_descriptors[method_idx - 1]:
                d = used_distance(db_descriptor, query_descriptor)
                if used_distance.__name__ in ["hellinger_kernel"]:
                    d = 1 / (d + 1e-16)
                distances.append(d)

            distances = np.array(distances)
            top_k_idx = np.argsort(distances)[:general_config.TOP_K_TEST]
            results.append(top_k_idx.tolist())

        # save pickle
        output_dir = io_config.RESULTS_DIR / io_config.TEST_NAME / f"method{method_idx}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "result.pkl"

        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        log.info(f"Saved results for method {method_idx} → {output_path}")
