"""
Color descriptors configuration.
Grid search generation + named configs for clarity.
"""

from itertools import product
from descriptors.color_descriptors import color_descriptors_func as descriptors
from utils import metrics

# Search space
COLOR_SPACES = {
    "gray": {
        "channels": [["Gray"]],
        "bins": [[256], [128], [64],[32],[16],[8]],
        "ranges": [[(0, 256)]],
    },
    "rgb": {
        "channels": [["B", "G", "R"]],
        "bins": [[256, 256, 256], [128, 128, 128], [64, 64, 64],[32,32,32],[16,16,16]],
        "ranges": [[(0, 256)] * 3],
    },
    "hsv": {
        "channels": [["H", "S", "V"]],
        "bins": [
            [180, 256, 256],   
            [90, 128, 128],    
            [45, 64, 64],
            [20,32,32],
            [10,16,16]    
        ],
        "ranges": [[(0, 180), (0, 256), (0, 256)]],
    },
    "lab": {
        "channels": [["L", "A", "B"]],
        "bins": [
            [256, 256, 256],
            [128, 128, 128],
            [64, 64, 64],
            [32,32,32]
        ],
        "ranges": [[(0, 256)] * 3],
    },
    "ycbcr": {
        "channels": [["Y", "Cr", "Cb"]],
        "bins": [
            [256, 256, 256],
            [128, 128, 128],
            [64, 64, 64],
            [32,32,32],
            [16,16,16]
        ],
        "ranges": [[(0, 256)] * 3],
    },
}

WEIGHTS_OPTIONS = {
    1: [[1.0], [0.8], [1.2]],
    2: [[1.0, 1.0], [0.8, 1.2], [1.2, 0.8]],
    3: [
        [1.0, 1.0, 1.0],
        [1.0, 0.8, 1.2],
        [1.2, 1.0, 0.8],
        [3.0,1.0,1.0],
        [0.5,3.0,1.0]
    ],
}

# Grid search configs

COLOR_DESCRIPTORS_CONFIGS = []
for space, params in COLOR_SPACES.items():
    for channels, bins, ranges in product(params["channels"], params["bins"], params["ranges"]):
        if len(channels) != len(ranges):
            continue
        for weights in WEIGHTS_OPTIONS[len(channels)]:
            name = f"{space}_{'_'.join(channels)}_bins{'-'.join(map(str, bins))}_w{'-'.join(map(str, weights))}"
            cfg = {
                "name": name,
                "color_space": space,
                "channels": channels,
                "bins": bins,
                "ranges": ranges,
                "weights": weights,
            }
            COLOR_DESCRIPTORS_CONFIGS.append(cfg)

CONFIGS_BY_NAME = {cfg["name"]: cfg for cfg in COLOR_DESCRIPTORS_CONFIGS}

# Build descriptors (functions)

INDIVIDUAL_COLOR_DESCRIPTORS = [
    descriptors.generic_color_descriptor(
        color_space=cfg["color_space"],
        channels=cfg["channels"],
        bins=cfg["bins"],
        ranges=cfg["ranges"],
        weights=cfg["weights"]
    )
    for cfg in COLOR_DESCRIPTORS_CONFIGS
]

INDIVIDUAL_COLOR_DESCRIPTORS_NAMES = [cfg["name"] for cfg in COLOR_DESCRIPTORS_CONFIGS]



# Mixed descriptors

MIXED_CONFIGS = {
    "mixed_gray_hs_lab": [
        CONFIGS_BY_NAME["gray_Gray_bins256_w1.0"],
        CONFIGS_BY_NAME["hsv_H_S_V_bins180-256-256_w1.0-1.0-1.0"],
        CONFIGS_BY_NAME["lab_L_A_B_bins256-256-256_w1.0-1.0-1.0"],
    ],
    "mixed_rgb_hsv": [
        CONFIGS_BY_NAME["rgb_B_G_R_bins256-256-256_w1.0-1.0-1.0"],
        CONFIGS_BY_NAME["hsv_H_S_V_bins180-256-256_w1.0-1.0-1.0"],
    ]
}

MIXED_COLOR_DESCRIPTORS = {
    name: descriptors.mixed_concat_descriptor([
        {
            "color_space": cfg["color_space"],
            "channels": cfg["channels"],
            "bins": cfg["bins"],
            "ranges": cfg["ranges"],
            "weights": cfg["weights"],
        }
        for cfg in cfgs
    ])
    for name, cfgs in MIXED_CONFIGS.items()
}



# Final lists

ALL_COLOR_DESCRIPTORS = INDIVIDUAL_COLOR_DESCRIPTORS + list(MIXED_COLOR_DESCRIPTORS.values())
ALL_COLOR_DESCRIPTORS_NAMES = INDIVIDUAL_COLOR_DESCRIPTORS_NAMES + list(MIXED_COLOR_DESCRIPTORS.keys())


PRECOMPUTED_COLOR_DESCRIPTORS = ALL_COLOR_DESCRIPTORS
PRECOMPUTED_COLOR_DESCRIPTOR_NAMES = ALL_COLOR_DESCRIPTORS_NAMES


DEV_COLOR_DESCRIPTORS = ALL_COLOR_DESCRIPTORS
DEV_COLOR_DESCRIPTOR_NAMES = ALL_COLOR_DESCRIPTORS_NAMES

PREDICT_COLOR_DESCRIPTORS = [
    descriptors.generic_color_descriptor(
        color_space=cfg["color_space"],
        channels=cfg["channels"],
        bins=cfg["bins"],
        ranges=cfg["ranges"],
        weights=cfg["weights"]
    )
    for cfg in [CONFIGS_BY_NAME["hsv_H_S_V_bins180-256-256_w1.0-1.0-1.0"], 
                CONFIGS_BY_NAME["lab_L_A_B_bins256-256-256_w1.0-1.0-1.0"]]
]

PREDICTING_COLOR_DESCRIPTORS = [
    (PREDICT_COLOR_DESCRIPTORS[0], metrics.canberra_distance),
    (PREDICT_COLOR_DESCRIPTORS[1], metrics.hellinger_kernel),
]