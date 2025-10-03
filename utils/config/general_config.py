from utils import metrics

DESCRIPTORS = ["COLOR_DESCRIPTORS"]

WANTED_DISTANCES = [
    metrics.euclidean_distance,
    metrics.x2_dist,
    metrics.bhattacharyya_distance,
    metrics.l1_distance,
    (metrics.histogram_intersection, 1),
    (metrics.hellinger_kernel, 1),
]