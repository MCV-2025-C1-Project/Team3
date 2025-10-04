from utils import metrics

# General execution parameters
PRECOMPUTE = False       # Whether to compute DB descriptors
DEV_PREDICTION = True   # Whether to run dev pipeline
TEST_PREDICTION = False # Whether to run test pipeline


TOP_K_TEST = 10
K_VALUES = [1,5]

# Descriptor families to use
DESCRIPTORS = ["COLOR_DESCRIPTORS"]

# Distance metrics
WANTED_DISTANCES = [
    metrics.euclidean_distance,
    metrics.x2_dist,
    metrics.bhattacharyya_distance,
    metrics.l1_distance,
    (metrics.histogram_intersection, 1),
    (metrics.hellinger_kernel, 1),
    metrics.earth_movers_distance,
    metrics.canberra_distance,
]
