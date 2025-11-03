from utils import global_metrics

# General execution parameters
PRECOMPUTE = True       # Whether to compute DB descriptors
DEV_PREDICTION = False   # Whether to run dev pipeline
TEST_PREDICTION = False # Whether to run test pipeline


TOP_K_TEST = 10
K_VALUES = [1,5]

# Descriptor families to use
DESCRIPTORS = [#"COLOR_DESCRIPTORS",
               #"TEXTURE_DESCRIPTORS",
               "KEYPOINT_DESCRIPTORS"
            ]

# Distance metrics
WANTED_DISTANCES = [
    global_metrics.euclidean_distance,
    global_metrics.x2_dist,
    global_metrics.bhattacharyya_distance,
    global_metrics.l1_distance,
    (global_metrics.histogram_intersection, 1),
    (global_metrics.hellinger_kernel, 1),
    global_metrics.earth_movers_distance,
    global_metrics.canberra_distance,
]

LOCAL_DISTANCES_NAMES = ["match_count", "match_normalized", "match_geometric"]

REMOVE_BACKGROUND = True
SAVE_BACKGROUND_MASK = False
APPLY_BACKGROUND_REMOVAL = True
