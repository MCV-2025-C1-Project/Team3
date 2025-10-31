"""
Color descriptors configuration.
Grid search generation + named configs for clarity.
"""

from itertools import product, combinations
from descriptors.texture_descriptors import texture_descriptors_func as descriptors
from utils import metrics

# Search space
TEXTURE_TECHNIQUES = {

    # "LBP": {
    #     "P": [4, 8, 16],
    #     "R": [
    #         1, 2, 3
    #     ],
    #     "method": ["default", "ror", "uniform", "nri_uniform"]
    # },

    "DCT": {
       "block_size": [4,],# 8, 16],
       "top_k": [
           30,#10, 20, 30
       ],
    },

    #"DWT": {
    #    "levels": [1, 2, 3],
    #    "divisions": [4, 8, 16],
    #    "top_k": [10, 20, 30],
    #    "wavelet": ["haar", "db2", "sym2", "coif1"],
    #},
}

# Grid search configs
TEXTURE_DESCRIPTORS_CONFIGS = []

for descriptor, params in TEXTURE_TECHNIQUES.items():
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
            "texture_descriptor": descriptor,
            "specific_parameters": specific_params,
        }

        TEXTURE_DESCRIPTORS_CONFIGS.append(cfg)

                

CONFIGS_BY_NAME = {cfg["name"]: cfg for cfg in TEXTURE_DESCRIPTORS_CONFIGS}

# Build descriptors (functions)

INDIVIDUAL_TEXTURE_DESCRIPTORS = [
    descriptors.generic_texture_descriptor(
        descriptor_type=cfg["texture_descriptor"],
        descriptor_specific_parameters=cfg["specific_parameters"],
    )
    for cfg in TEXTURE_DESCRIPTORS_CONFIGS
]

INDIVIDUAL_TEXTURE_DESCRIPTORS_NAMES = [cfg["name"] for cfg in TEXTURE_DESCRIPTORS_CONFIGS]


# Final lists
ALL_TEXTURE_DESCRIPTORS = INDIVIDUAL_TEXTURE_DESCRIPTORS
ALL_TEXTURE_DESCRIPTORS_NAMES = INDIVIDUAL_TEXTURE_DESCRIPTORS_NAMES


PRECOMPUTED_TEXTURE_DESCRIPTORS = ALL_TEXTURE_DESCRIPTORS
PRECOMPUTED_TEXTURE_DESCRIPTOR_NAMES = ALL_TEXTURE_DESCRIPTORS_NAMES


DEV_TEXTURE_DESCRIPTORS = ALL_TEXTURE_DESCRIPTORS
DEV_TEXTURE_DESCRIPTOR_NAMES = ALL_TEXTURE_DESCRIPTORS_NAMES

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
PREDICTING_TEXTURE_DESCRIPTORS = [(ALL_TEXTURE_DESCRIPTORS[0], metrics.canberra_distance)]
