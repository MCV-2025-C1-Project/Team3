from pathlib import Path

# Base directory
# Go 2 levels up to reach Team3 root
BASE_DIR = Path(__file__).resolve().parents[1]



# Data dir
DB_NAME = "BBDD"
DEV_NAME = "qsd1_w1"
TEST_NAME = "qst1_w1"

DATA_DIR = BASE_DIR / "data"
DB_DIR   = DATA_DIR / DB_NAME
DEV_DIR  = DATA_DIR / DEV_NAME
TEST_DIR = DATA_DIR / TEST_NAME

# Outputs
RESULTS_DIR = BASE_DIR / "results"
DESCRIPTORS_DIR = BASE_DIR / "descriptors"
COLOR_DESC_DIR  = DESCRIPTORS_DIR / "color_descriptors/stored_color_descriptors"

HIST_DIR        = RESULTS_DIR / "histograms"
STORE_HISTOGRAMS = False
STORE_RESULTS_TXT_BY_DESCRIPTOR = False

# Helpers
def ensure_dirs() -> None:
    """Create all required directories if they don’t exist."""
    for p in [RESULTS_DIR, DESCRIPTORS_DIR, HIST_DIR]:
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
