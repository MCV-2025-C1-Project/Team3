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

    # Collect active descriptor types
    if "COLOR_DESCRIPTORS" in general_config.DESCRIPTORS:
        from config.color_descriptors_config import PRECOMPUTED_COLOR_DESCRIPTORS
        ALL_BLOCKS["COLOR_DESCRIPTORS"] = {
            "descriptors": PRECOMPUTED_COLOR_DESCRIPTORS,
            "dir": io_config.COLOR_DESC_DIR
        }

    if "TEXTURE_DESCRIPTORS" in general_config.DESCRIPTORS:
        from config.texture_descriptors_config import PRECOMPUTED_TEXTURE_DESCRIPTORS
        ALL_BLOCKS["TEXTURE_DESCRIPTORS"] = {
            "descriptors": PRECOMPUTED_TEXTURE_DESCRIPTORS,
            "dir": io_config.TEXTURE_DESC_DIR
        }

    # Ensure output directories exist
    io_config.ensure_dirs()

    image_number = io_config.count_jpgs(io_config.DB_DIR)

    # Compute and save descriptors
    for block, data in ALL_BLOCKS.items():
        data["dir"].mkdir(parents=True, exist_ok=True)
        names = [f.__name__ for f in data["descriptors"]]

        # For each descriptor type, open a separate HDF5 file
        for name, function in zip(names, data["descriptors"]):
            file_path = data["dir"] / f"{name}.h5"

            with h5py.File(file_path, "w") as f:
                first_image_path = io_config.db_image_path(0)
                first_img = cv2.imread(first_image_path)
                first_img = cv2.resize(first_img, (256, 256))

                # Get descriptor shape from first image
                first_descriptor = function(first_img, io_config.DB_NAME, 0, visualize=False)
                descriptor_size = first_descriptor.shape[0]

                # Create dataset once
                dset = f.create_dataset(
                    "descriptors",
                    shape=(image_number, descriptor_size),
                    dtype=np.float64,
                    compression="gzip"
                )

                # Store first descriptor
                dset[0] = first_descriptor

                # Process the rest of the images
                for i in tqdm(range(1, image_number), desc=f"Processing {name}"):
                    image_path = io_config.db_image_path(i)
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (256, 256))

                    descriptor = function(
                        img,
                        io_config.DB_NAME,
                        i,
                        visualize=io_config.STORE_HISTOGRAMS
                    )

                    dset[i] = descriptor


            