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
        "bins": [[256]],
        "ranges": [[(0, 256)]],
    },
    "rgb": {
        "channels": [["B", "G", "R"]],
        "bins": [[256, 256, 256], [128, 128, 128]],
        "ranges": [[(0, 256)] * 3],
    },
    "hsv": {
        "channels": [["H", "S", "V"], ["H", "S"]],
        "bins": [[180, 256, 256], [180, 100, 32], [180, 256]],
        "ranges": [
            [(0, 180), (0, 256), (0, 256)],
            [(0, 180), (0, 256)],
        ],
    },
    "lab": {
        "channels": [["L", "A", "B"]],
        "bins": [[256, 256, 256], [128, 128, 128]],
        "ranges": [[(0, 256)] * 3],
    },
    "ycbcr": {
        "channels": [["Y", "Cr", "Cb"]],
        "bins": [[256, 256, 256]],
        "ranges": [[(0, 256)] * 3],
    },
}

WEIGHTS_OPTIONS = {
    1: [[1.0]],
    2: [[1.0,1.0]],
    3: [
        [1.0, 1.0, 1.0],
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5],
    ],
}

# Grid search configs

COLOR_DESCRIPTORS_CONFIGS = []
for space, params in COLOR_SPACES.items():
    for channels, bins, ranges in product(params["channels"], params["bins"], params["ranges"]):
        if len(channels) != len(ranges):
            continue
        for weights in WEIGHTS_OPTIONS[len(channels)]:
            name = f"{space}_{'-'.join(channels)}_bins{'-'.join(map(str, bins))}_w{'-'.join(map(str, weights))}"
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
        CONFIGS_BY_NAME["hsv_H-S_bins180-256_w1.0-1.0"],
        CONFIGS_BY_NAME["lab_L-A-B_bins256-256-256_w1.0-1.0-1.0"],
    ],
    "mixed_rgb_hsv": [
        CONFIGS_BY_NAME["rgb_B-G-R_bins256-256-256_w1.0-1.0-1.0"],
        CONFIGS_BY_NAME["hsv_H-S-V_bins180-256-256_w1.0-1.0-1.0"],
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


PREDICTING_COLOR_DESCRIPTORS = [
    (
        descriptors.generic_color_descriptor(
            color_space=CONFIGS_BY_NAME["gray_Gray_bins256_w1.0"]["color_space"],
            channels=CONFIGS_BY_NAME["gray_Gray_bins256_w1.0"]["channels"],
            bins=CONFIGS_BY_NAME["gray_Gray_bins256_w1.0"]["bins"],
            ranges=CONFIGS_BY_NAME["gray_Gray_bins256_w1.0"]["ranges"],
            weights=CONFIGS_BY_NAME["gray_Gray_bins256_w1.0"]["weights"],
        ),
        metrics.canberra_distance,
    ),
    (
        descriptors.generic_color_descriptor(
            color_space=CONFIGS_BY_NAME["gray_Gray_bins256_w1.0"]["color_space"],
            channels=CONFIGS_BY_NAME["gray_Gray_bins256_w1.0"]["channels"],
            bins=CONFIGS_BY_NAME["gray_Gray_bins256_w1.0"]["bins"],
            ranges=CONFIGS_BY_NAME["gray_Gray_bins256_w1.0"]["ranges"],
            weights=CONFIGS_BY_NAME["gray_Gray_bins256_w1.0"]["weights"],
        ),
        metrics.hellinger_kernel,
    ),
]