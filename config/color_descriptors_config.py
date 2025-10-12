"""
Color descriptors configuration.
Grid search generation + named configs for clarity.
"""

from itertools import product, combinations
from descriptors.color_descriptors import color_descriptors_func as descriptors
from utils import metrics

# Search space
COLOR_SPACES = {

    #"gray": {
    #    "channels": [["Gray"]],
    #    "bins": [[256], [128], [64],[32],[16],[8]],
    #    "ranges": [[(0, 256)]],
    #},
    #"rgb": {
    #    "channels": [["B", "G", "R"]],
    #    "bins": [[256, 256, 256], [128, 128, 128], [64, 64, 64],[32,32,32],[16,16,16]],
    #    "ranges": [[(0, 256)] * 3],
    #},
    # "lab": {
    #    "channels": [["L", "A", "B"]],
    #    "bins": [
    #     #    [256, 256, 256],
    #     #    [128, 128, 128],
    #     #    [64, 64, 64],
    #     #    [32,32,32]
    #          [16, 16, 16],
    #          [8, 8, 8],
    #          [4, 4, 4],
    #          [2, 2, 2]
    #    ],
    #    "ranges": [[(0, 256)] * 3],
    # },
    
    # "hsv": {
    #     "channels": [["H", "S", "V"]],
    #     "bins": [
    #         #[180, 256, 256],   
    #         #[90, 128, 128],    
    #         #[45, 64, 64],
    #         # [20,32,32],
    #         # [10,16,16],
    #         [4, 4, 4],
    #         [2, 2, 2]
    #     ],
    #     "ranges": [[(0, 180), (0, 256), (0, 256)]],
    # },
    "ycbcr": {
        "channels": [["Y", "Cr", "Cb"]],
        "bins": [
            #[256, 256, 256],
            [128, 128, 128],
            [64, 64, 64],
            [32,32,32],
            [16, 16, 16],
            # [8, 8, 8],
            # [4, 4, 4],
            # [2, 2, 2]
        ],
        "ranges": [[(0, 256)] * 3],
    },
}

WEIGHTS_OPTIONS = {
    1: [
        [1.0], 
        [0.8], 
        [1.2]
    ],
    2: [
        #[1.0, 1.0], 
        [0.8, 1.2], 
        #[1.2, 0.8]
    ],
    3: [
        # [1.0, 1.0, 1.0],
        # [1.0, 0.8, 1.2],
        # [1.2, 1.0, 0.8],
        # [3.0,1.0,1.0],
        [0.5,3.0,1.0]
    ],
}

HIERARCHICAL_LEVELS = [
    [10], [8], [7], [1, 5, 7], [5, 7], [1, 7], [8, 10]
    # [10], [9, 10], [10, 11], [9, 10, 11], [9, 10, 11, 12]
]

# Grid search configs

COLOR_DESCRIPTORS_CONFIGS = []
for space, params in COLOR_SPACES.items():
    for channels, bins, ranges in product(params["channels"], params["bins"], params["ranges"]):
        if len(channels) != len(ranges):
            continue
        for weights in WEIGHTS_OPTIONS[len(channels)]:
            for hierarchical_levels in HIERARCHICAL_LEVELS:
                name = f"{space}_{'_'.join(channels)}_bins{'-'.join(map(str, bins))}_w{'-'.join(map(str, weights))}_hier{'-'.join(map(str, hierarchical_levels))}"
                cfg = {
                    "name": name,
                    "color_space": space,
                    "channels": channels,
                    "bins": bins,
                    "ranges": ranges,
                    "weights": weights,
                    "hierarchical": hierarchical_levels,
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
        weights=cfg["weights"],
        hierarchical_levels=cfg["hierarchical"]
    )
    for cfg in COLOR_DESCRIPTORS_CONFIGS
]

INDIVIDUAL_COLOR_DESCRIPTORS_NAMES = [cfg["name"] for cfg in COLOR_DESCRIPTORS_CONFIGS]


# 2D combinations of channels
COLOR_DESCRIPTORS_2D_CONFIGS = []
for space, params in COLOR_SPACES.items():
    for channels_all, bins_all, ranges_all in product(params["channels"], params["bins"], params["ranges"]):
        # channels_all is a list like ["H","S","V"]; create all 2-channel pairs
        if len(channels_all) < 2:
            continue
        # iterate over index pairs to pick corresponding bins/ranges
        for idx_pair in combinations(range(len(channels_all)), 2):
            pair_channels = [channels_all[i] for i in idx_pair]
            pair_bins = [bins_all[i] for i in idx_pair]
            pair_ranges = [ranges_all[i] for i in idx_pair]
            for weights in WEIGHTS_OPTIONS[len(pair_channels)]:
                for hierarchical_levels in HIERARCHICAL_LEVELS:
                    name = f"{space}_{'_'.join(pair_channels)}_bins{'-'.join(map(str, pair_bins))}_w{'-'.join(map(str, weights))}_hier{'-'.join(map(str, hierarchical_levels))}"
                    cfg = {
                        "name": name,
                        "color_space": space,
                        "channels": pair_channels,
                        "bins": pair_bins,
                        "ranges": pair_ranges,
                        "weights": weights,
                        "hierarchical": hierarchical_levels,
                    }
                    COLOR_DESCRIPTORS_2D_CONFIGS.append(cfg)

CONFIGS_2D_BY_NAME = {cfg["name"]: cfg for cfg in COLOR_DESCRIPTORS_2D_CONFIGS}

INDIVIDUAL_COLOR_DESCRIPTORS_2D = [
    descriptors.generic_color_descriptor_2d(
        color_space=cfg["color_space"],
        channels=cfg["channels"],
        bins=cfg["bins"],
        ranges=cfg["ranges"],
        weights=cfg["weights"],
        hierarchical_levels=cfg["hierarchical"]
    )
    for cfg in COLOR_DESCRIPTORS_2D_CONFIGS
]

INDIVIDUAL_COLOR_DESCRIPTORS_2D_NAMES = [cfg["name"] for cfg in COLOR_DESCRIPTORS_2D_CONFIGS]


# 3D combinations of channels
COLOR_DESCRIPTORS_3D_CONFIGS = []
for space, params in COLOR_SPACES.items():
    for channels, bins, ranges in product(params["channels"], params["bins"], params["ranges"]):
        if len(channels) != 3:
            continue
        if len(channels) != len(ranges):
            continue
        for weights in WEIGHTS_OPTIONS[len(channels)]:
            for hierarchical_levels in HIERARCHICAL_LEVELS:
                name = f"{space}_{'_'.join(channels)}_bins{'-'.join(map(str, bins))}_w{'-'.join(map(str, weights))}_hier{'-'.join(map(str, hierarchical_levels))}"
                cfg = {
                    "name": name,
                    "color_space": space,
                    "channels": channels,
                    "bins": bins,
                    "ranges": ranges,
                    "weights": weights,
                    "hierarchical": hierarchical_levels,
                }
                COLOR_DESCRIPTORS_3D_CONFIGS.append(cfg)

CONFIGS_3D_BY_NAME = {cfg["name"]: cfg for cfg in COLOR_DESCRIPTORS_3D_CONFIGS}

INDIVIDUAL_COLOR_DESCRIPTORS_3D = [
    descriptors.generic_color_descriptor_3d(
        color_space=cfg["color_space"],
        channels=cfg["channels"],
        bins=cfg["bins"],
        ranges=cfg["ranges"],
        weights=cfg["weights"],
        hierarchical_levels=cfg["hierarchical"]
    )
    for cfg in COLOR_DESCRIPTORS_3D_CONFIGS
]

INDIVIDUAL_COLOR_DESCRIPTORS_3D_NAMES = [cfg["name"] for cfg in COLOR_DESCRIPTORS_3D_CONFIGS]




# Final lists
ALL_COLOR_DESCRIPTORS = INDIVIDUAL_COLOR_DESCRIPTORS_2D
ALL_COLOR_DESCRIPTORS_NAMES = INDIVIDUAL_COLOR_DESCRIPTORS_2D_NAMES


PRECOMPUTED_COLOR_DESCRIPTORS = ALL_COLOR_DESCRIPTORS
PRECOMPUTED_COLOR_DESCRIPTOR_NAMES = ALL_COLOR_DESCRIPTORS_NAMES


DEV_COLOR_DESCRIPTORS = ALL_COLOR_DESCRIPTORS
DEV_COLOR_DESCRIPTOR_NAMES = ALL_COLOR_DESCRIPTORS_NAMES

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
PREDICTING_COLOR_DESCRIPTORS = [(ALL_COLOR_DESCRIPTORS[0], metrics.earth_movers_distance)]
