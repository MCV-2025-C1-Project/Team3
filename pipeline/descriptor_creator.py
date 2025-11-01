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

    if "KEYPOINT_DESCRIPTORS" in general_config.DESCRIPTORS:
        from config.keypoint_descriptors_config import PRECOMPUTED_KEYPOINT_DESCRIPTORS
        ALL_BLOCKS["KEYPOINT_DESCRIPTORS"] = {
            "descriptors": PRECOMPUTED_KEYPOINT_DESCRIPTORS,
            "dir": io_config.KEYPOINT_DESC_DIR
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
                # create a group-level index attribute with number of images
                f.attrs["num_images"] = image_number

                for i in tqdm(range(image_number), desc=f"Computing {name}"):
                    img_path = io_config.db_image_path(i)
                    img = cv2.imread(img_path)

                    desc_info = function(img, io_config.DB_NAME, i, visualize=False)
                    gname = f"img_{i:05d}"
                    grp = f.create_group(gname)

                    if desc_info["type"] == "local":
                        kp_arr = desc_info.get("keypoints", np.zeros((0, 4), dtype=np.float32)).astype(np.float32)
                        desc_arr = desc_info.get("descriptors", np.zeros((0, 128), dtype=np.float32)).astype(np.float32)

                        # Save datasets (variable first dimension)
                        grp.create_dataset("keypoints", data=kp_arr, compression="gzip")
                        grp.create_dataset("descriptors", data=desc_arr, compression="gzip")

                    elif desc_info["type"] == "global":
                        desc_arr = np.asarray(desc_info["descriptors"], dtype=np.float32)
                        # store single dataset 'descriptors' per image for consistency
                        grp.create_dataset("descriptors", data=desc_arr, compression="gzip")

                    else:
                        raise ValueError(f"Unknown descriptor type: {desc_info.get('type')}")
            print(f"[precompute] saved {file_path}")


            