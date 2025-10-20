import numpy as np


def load_precomputed_descriptors(files):
    """Load precomputed descriptors from BBDD (txt files)."""
    precomputed_descriptors = []
    for file in files:
        images_descriptors = file['descriptors'][:, :]
        precomputed_descriptors.append(images_descriptors)
    return precomputed_descriptors