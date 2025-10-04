from descriptors.color_descriptors import color_descriptors_func as descriptors
from utils import metrics

COLOR_DESCRIPTORS_CONFIGS = [
    {"color_space": "gray",
     "channels": ["Gray"],
     "bins": [256],
     "ranges": [(0,256)],
     "weights": [1.0]},
    {"color_space": "rgb",
     "channels": ["B","G","R"],
     "bins": [256,256,256],
     "ranges": [(0,256)]*3,
     "weights": [1.0,1.0,1.0]},
    {"color_space": "hsv",
     "channels": ["H","S","V"],
     "bins": [180,256,256],
     "ranges": [(0,180),(0,256),(0,256)],
     "weights": [1.0,1.0,1.0]},
    {"color_space": "lab",
     "channels": ["L","A","B"],
     "bins": [256,256,256],
     "ranges": [(0,256)]*3,
     "weights": [1.0,1.0,1.0]},
    {"color_space": "ycbcr",
     "channels": ["Y","Cr","Cb"],
     "bins": [256,256,256],
     "ranges": [(0,256)]*3,
     "weights": [1.0,1.0,1.0]},
    {"color_space": "hsv",
     "channels": ["H","S"],
     "bins": [180,256],
     "ranges": [(0,180),(0,256)],
     "weights": [1.0,1.0]},
    {"color_space": "hsv",
     "channels": ["H","S","V"],
     "bins": [180,100,32],
     "ranges": [(0,180),(0,256),(0,256)],
     "weights": [1.0,1.0,1.0]},
]

# Build individual descriptors directly
INDIVIDUAL_COLOR_DESCRIPTORS = [
    descriptors.generic_color_descriptor(**cfg) for cfg in COLOR_DESCRIPTORS_CONFIGS
]

# Example of a mixed descriptor
MIXED_CONFIGS = [
    COLOR_DESCRIPTORS_CONFIGS[0],
    COLOR_DESCRIPTORS_CONFIGS[5],
    COLOR_DESCRIPTORS_CONFIGS[3],
]


MIXED_COLOR_CONCAT_DESCRIPTOR = descriptors.mixed_concat_descriptor(MIXED_CONFIGS)


# Final list
WANTED_COLOR_DESCRIPTORS = INDIVIDUAL_COLOR_DESCRIPTORS + [MIXED_COLOR_CONCAT_DESCRIPTOR]


WANTED_COLOR_DESCRIPTORS_NAMES = [f"{f["color_space"]}_{f['bins']}" for f in COLOR_DESCRIPTORS_CONFIGS] + ["Mixed_Concat", "Mixed_Sum"]

# Predicting list
PREDICTING_COLOR_DESCRIPTORS = [(INDIVIDUAL_COLOR_DESCRIPTORS[2], metrics.canberra_distance),
                                (INDIVIDUAL_COLOR_DESCRIPTORS[3], metrics.hellinger_kernel)]