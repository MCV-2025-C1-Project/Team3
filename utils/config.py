from pathlib import Path
from descriptors.color_descriptors import color_descriptors_func as descriptors
from utils import metrics

# Base directory
BASE_DIR = Path(__file__).resolve().parents[1]


# Data dir
DB_NAME = "BBDD"
DEV_NAME = "qsd1_w1"
DATA_DIR = BASE_DIR / "data"
DB_DIR   = DATA_DIR / DB_NAME
DEV_DIR  = DATA_DIR / DEV_NAME

# Outputs
RESULTS_DIR = BASE_DIR / "results"
DESCRIPTORS_DIR = BASE_DIR / "descriptors"
COLOR_DESC_DIR  = DESCRIPTORS_DIR / "color_descriptors"
HIST_DIR        = RESULTS_DIR / "histograms"

# Retrieval parameters
DESCRIPTORS_TYPE_DIR = COLOR_DESC_DIR
STORE_HISTOGRAMS = True
TOP_K = 5
WANTED_DESCRIPTORS_OFFLINE = [
    descriptors.gray_descriptor,
    descriptors.rgb_descriptor,
    descriptors.hsv_descriptor,
]

WANTED_DESCRIPTORS_ONLINE = [
    descriptors.gray_descriptor,
    descriptors.rgb_descriptor,
    descriptors.hsv_descriptor,
]

WANTED_DISTANCES = [
    metrics.euclidean_distance,
    metrics.x2_dist,
    metrics.bhattacharyya_distance,
    metrics.l1_distance,
    (metrics.histogram_intersection, 1),
    (metrics.hellinger_kernel, 1),
]

# Helpers
def ensure_dirs() -> None:
    """Create all required directories if they don’t exist."""
    for p in [RESULTS_DIR, DESCRIPTORS_DIR, COLOR_DESC_DIR, HIST_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def db_image_path(i: int) -> Path:
    """Path to a DB image."""
    return DB_DIR / f"bbdd_{i:05d}.jpg"

def dev_image_path(i: int) -> Path:
    """Path to a development set image."""
    return DEV_DIR / f"{i:05d}.jpg"

def count_jpgs(folder: Path) -> int:
    """Count only .jpg files in a folder (avoids .pkl, etc.)."""
    return len([p for p in folder.iterdir() if p.suffix.lower() == ".jpg"])
