import cv2
import numpy as np
import pickle
from config import general_config
from config import io_config
from config.color_descriptors_config import PREDICTING_COLOR_DESCRIPTORS
from utils.common import load_precomputed_descriptors



def predict_and_save_results():
    """
    Compute predictions for all test queries and save results as PKL files.
    One PKL per method in PREDICTING_COLOR_DESCRIPTORS.

    Output files:
    results/method1/result.pkl
    results/method2/result.pkl
    """
    images = sorted([p for p in io_config.TEST_DIR.iterdir() if p.suffix.lower() == ".jpg"])

    descriptors_names = [f[0].__name__ for f in PREDICTING_COLOR_DESCRIPTORS]
    files = [(io_config.COLOR_DESC_DIR / f"{name}.txt").open("r") for name in descriptors_names]
    precomputed_descriptors = load_precomputed_descriptors(files)

    for method_idx, (used_descriptor, used_distance) in enumerate(PREDICTING_COLOR_DESCRIPTORS, start=1):
        print(f"Running prediction for method {method_idx}: {used_descriptor.__name__} + {used_distance.__name__}")

        results = []
        for image in images:
            img = cv2.imread(str(image))
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
        output_dir = io_config.RESULTS_DIR / f"method{method_idx}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "result.pkl"

        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        print(f"Saved results for method {method_idx} → {output_path}")
