"""
Pipeline to precompute descriptors for the full DB (BBDD).
"""

import cv2
import numpy as np
from config import io_config, general_config
from tqdm import tqdm
import h5py


def precompute_descriptors():
    """
    Compute and save descriptors for the full DB (BBDD).
    Uses the list of descriptors defined in general_config.DESCRIPTORS.
    """
    ALL_BLOCKS = {}

    if "COLOR_DESCRIPTORS" in general_config.DESCRIPTORS:
        from config.color_descriptors_config import PRECOMPUTED_COLOR_DESCRIPTORS
        ALL_BLOCKS["COLOR_DESCRIPTORS"] = {
            "descriptors": PRECOMPUTED_COLOR_DESCRIPTORS,
            "dir": io_config.COLOR_DESC_DIR
        }

    # ADD MORE TYPES IN FUTURE
    # Ensure dirs
    io_config.ensure_dirs()

    # Open files per descriptor
    for block, data in ALL_BLOCKS.items():
        data["dir"].mkdir(parents=True, exist_ok=True)
        names = [f.__name__ for f in data["descriptors"]]
        data["files"] = [h5py.File((data["dir"] / f"{name}.h5"), 'w') for name in names]

    # Compute
    for i in tqdm(range(io_config.count_jpgs(io_config.DB_DIR)), desc="Precomputed images:"):
        image_path = io_config.db_image_path(i)
        img = cv2.imread(image_path)
        for block, data in ALL_BLOCKS.items():
            descriptors = []
            for idx, function in enumerate(data["descriptors"]):
                descriptor = function(img, io_config.DB_NAME, i, visualize=io_config.STORE_HISTOGRAMS)
                if 'descriptors' not in data["files"][idx]:
                    data['files'][idx].create_dataset('descriptors', shape=(0, descriptor.shape[0]),
                                                      maxshape=(None, descriptor.shape[0]),
                                                      compression='gzip')
                dataset = data["files"][idx]['descriptors']
                dataset.resize(dataset.shape[0] + 1, axis=0)
                dataset[-1] = descriptor

    # Close files
    for block, data in ALL_BLOCKS.items():
        for f in data["files"]:
            f.close()
