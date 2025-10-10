import numpy as np

def load_precomputed_descriptors(files):
    """Load precomputed descriptors from BBDD (txt files)."""
    precomputed_descriptors = []
    for file in files:
        images_descriptors = [np.fromstring(line, sep=" ") for line in file]
        file.close()
        precomputed_descriptors.append(images_descriptors)
    return precomputed_descriptors