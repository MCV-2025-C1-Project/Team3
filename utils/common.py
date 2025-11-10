import numpy as np

def load_precomputed_descriptors(files):
    """
    Load precomputed descriptors from one or several HDF5 files.
    Supports both:
      - global descriptors (direct 'descriptors' dataset)
      - local keypoint descriptors (per-image groups)
    """
    precomputed_descriptors = []

    for f in files:
        if "descriptors" in f:
            images_descriptors = np.array(f["descriptors"][:])
            precomputed_descriptors.append(images_descriptors)
        else:
            images_descriptors = []
            for key in f.keys():
                if not str(key).startswith("img_"):
                    continue

                grp = f[key]
                if "descriptors" in grp and "keypoints" in grp:
                    desc_entry = {
                        "type": "local",
                        "keypoints": np.array(grp["keypoints"], dtype=np.float32),
                        "descriptors": np.array(grp["descriptors"])
                    }
                    images_descriptors.append(desc_entry)
                elif "descriptors" in grp:
                    desc_entry = {
                        "type": "global",
                        "descriptors": np.array(grp["descriptors"], dtype=np.float32)
                    }
                    images_descriptors.append(desc_entry)

            precomputed_descriptors.append(images_descriptors)

    return precomputed_descriptors
