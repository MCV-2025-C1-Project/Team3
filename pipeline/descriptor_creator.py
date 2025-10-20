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


    image_number = io_config.count_jpgs(io_config.DB_DIR)
    # Compute
    for i in tqdm(range(image_number), desc="Precomputed images:"):
        image_path = io_config.db_image_path(i)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (256, 256))
        for block, data in ALL_BLOCKS.items():
            descriptors = []
            for idx, function in enumerate(data["descriptors"]):
                descriptor = function(img, io_config.DB_NAME, i, visualize=io_config.STORE_HISTOGRAMS)
                if 'descriptors' not in data["files"][idx]:
                    data['files'][idx].create_dataset('descriptors', shape=(image_number, descriptor.shape[0]),
                                                      maxshape=(image_number, descriptor.shape[0]),
                                                      dtype=np.float64,
                                                      compression='gzip')
                dataset = data["files"][idx]['descriptors']
                dataset[i] = descriptor

    # Close files
    for block, data in ALL_BLOCKS.items():
        for f in data["files"]:
            f.close()