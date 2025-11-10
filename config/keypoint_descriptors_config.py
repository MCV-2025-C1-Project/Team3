"""
Color descriptors configuration.
Grid search generation + named configs for clarity.
"""

from itertools import product, combinations
import descriptors.keypoint_descriptors.keypoint_descriptors_func as descriptors
from utils import global_metrics
from utils.local_metrics import match_count, match_geometric,match_reciprocal_count
from pipeline.descriptor_creator import sanitize_filename

# Search space
KEYPOINT_TECHNIQUES = {

    "sift": {
         "sigma": [1.2, 
                   1.8,
                   2.0,
                 ],
       "nfeatures": [500],
        "contrast_threshold": [0.01,0.04,0.06]
    },
    "orb": {
       "nfeatures": [500, 
                   ],
        "fastThreshold": [15, 20,25]

    },
    "color_sift": {
        "sigma": [
            1.2,
            1.8,
            2.0
        ],
        "nfeatures": [
            500,
        ],
        "contrast_threshold": [0.01,0.04,0.06]
    },
}

LOCAL_DISTANCES = [
    match_count,
    match_geometric,
    match_reciprocal_count,
]


DISCARDING_TYPES = [
    {"type": "threshold",
     "thresholds": [1,5,7,10,12,15,17,20, 30, 50, 100]},
]




# Grid search configs
KEYPOINT_DESCRIPTORS_CONFIGS = []

for descriptor, params in KEYPOINT_TECHNIQUES.items():
    param_names = list(params.keys())
    param_values = list(params.values())

    # Generate all combinations of parameter values
    for combination in product(*param_values):
        # Build dict of parameter name → selected value
        specific_params = dict(zip(param_names, combination))

        # Build part of name dynamically
        param_name_parts = []
        for i, (k, v) in enumerate(specific_params.items(), start=1):
            param_name_parts.append(f"param{i}{str(v)}")
            
            
        name = (
            f"{descriptor}_"
            + "_".join(param_name_parts)
        )


        cfg = {
            "name": name,
            "keypoint_descriptor": descriptor,
            "specific_parameters": specific_params,
        }

        KEYPOINT_DESCRIPTORS_CONFIGS.append(cfg)

                

CONFIGS_BY_NAME = {cfg["name"]: cfg for cfg in KEYPOINT_DESCRIPTORS_CONFIGS}

# Build descriptors (functions)

INDIVIDUAL_KEYPOINT_DESCRIPTORS = [
    descriptors.generic_keypoint_descriptor(
        descriptor_type=cfg["keypoint_descriptor"],
        descriptor_specific_parameters=cfg["specific_parameters"],
    )
    for cfg in KEYPOINT_DESCRIPTORS_CONFIGS
]


INDIVIDUAL_KEYPOINT_DESCRIPTORS_NAMES = [
    sanitize_filename(cfg["name"]) for cfg in KEYPOINT_DESCRIPTORS_CONFIGS
]


# Final lists
ALL_KEYPOINT_DESCRIPTORS = INDIVIDUAL_KEYPOINT_DESCRIPTORS
ALL_KEYPOINT_DESCRIPTORS_NAMES = INDIVIDUAL_KEYPOINT_DESCRIPTORS_NAMES


PRECOMPUTED_KEYPOINT_DESCRIPTORS = ALL_KEYPOINT_DESCRIPTORS
PRECOMPUTED_KEYPOINT_DESCRIPTOR_NAMES = ALL_KEYPOINT_DESCRIPTORS_NAMES


DEV_KEYPOINT_DESCRIPTORS = ALL_KEYPOINT_DESCRIPTORS
DEV_KEYPOINT_DESCRIPTOR_NAMES = ALL_KEYPOINT_DESCRIPTORS_NAMES

'''
PREDICT_COLOR_DESCRIPTORS = [
    descriptors.generic_color_descriptor(
        color_space=cfg["color_space"],
        channels=cfg["channels"],
        bins=cfg["bins"],
        ranges=cfg["ranges"],
        weights=cfg["weights"],
        hierarchical_levels=cfg["hierarchical"]
    )
    for cfg in [CONFIGS_BY_NAME["hsv_H_S_V_bins20-32-32_w1.2-1.0-0.8"], 
                CONFIGS_BY_NAME["ycbcr_Y_Cr_Cb_bins128-128-128_w0.5-3.0-1.0"]]
]
PREDICTING_COLOR_DESCRIPTORS = [
    (PREDICT_COLOR_DESCRIPTORS[0], metrics.canberra_distance),
    (PREDICT_COLOR_DESCRIPTORS[1], metrics.hellinger_kernel),
]
'''
PREDICTING_KEYPOINT_DESCRIPTORS = [(ALL_KEYPOINT_DESCRIPTORS[0], match_geometric)]
