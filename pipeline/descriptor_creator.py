"""
Pipeline to precompute descriptors for the full DB (BBDD).
"""

import cv2
import numpy as np
from config import io_config, general_config


def precompute_descriptors():
    """
    Compute and save descriptors for the full DB (BBDD).
    Uses the list of descriptors defined in general_config.DESCRIPTORS.
    """
    ALL_BLOCKS = {}

    if "COLOR_DESCRIPTORS" in general_config.DESCRIPTORS:
        from config.color_descriptors_config import WANTED_COLOR_DESCRIPTORS
        ALL_BLOCKS["COLOR_DESCRIPTORS"] = {
            "descriptors": WANTED_COLOR_DESCRIPTORS,
            "dir": io_config.DESCRIPTORS_DIR / "color_descriptors"
        }

    # ADD MORE TYPES IN FUTURE
    # Ensure dirs
    io_config.ensure_dirs()

    # Open files per descriptor
    for block, data in ALL_BLOCKS.items():
        data["dir"].mkdir(parents=True, exist_ok=True)
        names = [f.__name__ for f in data["descriptors"]]
        data["files"] = [(data["dir"] / f"{name}.txt").open("w") for name in names]

    # Compute
    for i in range(io_config.count_jpgs(io_config.DB_DIR)):
        image_path = io_config.db_image_path(i)
        img = cv2.imread(image_path)
        for block, data in ALL_BLOCKS.items():
            for idx, function in enumerate(data["descriptors"]):
                descriptor = function(img, io_config.DB_NAME, i, visualize=io_config.STORE_HISTOGRAMS)
                np.savetxt(data["files"][idx], descriptor[None])
        print(f"Processed {i}")

    # Close files
    for block, data in ALL_BLOCKS.items():
        for f in data["files"]:
            f.close()
