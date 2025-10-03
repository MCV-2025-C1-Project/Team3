
from config.color_descriptors_config import WANTED_COLOR_DESCRIPTORS_NAMES, WANTED_COLOR_DESCRIPTORS
from utils import metrics
import cv2
import logging
import logging.config
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from config import general_config,io_config

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


ALL_DESCRIPTORS = []

if "COLOR_DESCRIPTORS" in general_config.DESCRIPTORS:
    ALL_DESCRIPTORS.extend(WANTED_COLOR_DESCRIPTORS)

# EXTEND IN THE FUTURE

WANTED_DESCRIPTORS = ALL_DESCRIPTORS
WANTED_DISTANCES = general_config.WANTED_DISTANCES

# Metadata from io_config
NUMBER_IMAGE_DEV = io_config.count_jpgs(io_config.DEV_DIR)
NAME_OF_DEV_SET = io_config.DEV_NAME
K = io_config.TOP_K

# Names for logging and saving
descriptors_names = [f.__name__ for f in WANTED_DESCRIPTORS]
distances_names = [
    d[0].__name__ if isinstance(d, tuple) else d.__name__
    for d in WANTED_DISTANCES
]

files = []
for name in descriptors_names:
    if "COLOR_DESCRIPTORS" in general_config.DESCRIPTORS:
        files.append((io_config.COLOR_DESC_DIR / f"{name}.txt").open("r"))






#---------------------------------


def setup_logging():
    """
    Setup logging configuration from .ini file
    """

    logging.config.fileConfig("utils/logging.ini", disable_existing_loggers=False)
    
def compute_development_descriptors() -> list:
    """
    Computes the descriptors of the development set
    """

    all_descriptors = []
    
    log.info("Computing all the descriptor for development images")
    
    #For each image in the development folder, computes the specified descriptors
    for i in range(NUMBER_IMAGE_DEV):
        image_path = io_config.dev_image_path(i)
        img = cv2.imread(image_path)
        image_descriptors = []
        for function in WANTED_DESCRIPTORS:
            descriptor = function(img, NAME_OF_DEV_SET, i, visualize=False)
            image_descriptors.append(descriptor)
        all_descriptors.append(image_descriptors)
        
    
    log.info("Descriptors for development images computed")
    
    return all_descriptors

def load_precomputed_descriptors() -> list:
    """
    Loads the precomputated descriptors from the BBDD dataset
    """

    log.info("Loading precomputed descriptors of BBDD images")
    
    precomputed_descriptors = []
    
    #Reading of precomputed descriptors on the BBDD
    for file in files:
        images_descriptors = []
        for line in file:
            arr = np.fromstring(line, sep=" ")
            images_descriptors.append(arr)
        file.close()
        
        precomputed_descriptors.append(images_descriptors)
        
    log.info("Loaded")

    return precomputed_descriptors

def compute_distances(all_descriptors : list, precomputed_descriptors : list) -> list:
    """
    Computes the distances between the development and BBDD datasets
    """

    log.info("Computing distances between each development and BBDD image")
    
    all_metrics = []
    
    for objective_image in all_descriptors:
        image_metrics = []
        for idx, descriptor in enumerate(objective_image):
            found_metrics = []
            objective_descriptors = precomputed_descriptors[idx]
            for distance_function in WANTED_DISTANCES:
                if isinstance(distance_function, tuple):
                    distance_function = distance_function[0]
                distances = []
                for objective_descriptor in objective_descriptors:
                    distances.append(distance_function(descriptor, objective_descriptor))
                found_metrics.append(distances)
            image_metrics.append(found_metrics)
            
        all_metrics.append(image_metrics)
        

    log.info("Computed")
    
    return all_metrics

def write_results(all_metrics, ground_truth : list):
    """
    Writes the results on a txt. One per descriptor used
    """

    log.info("Outputting results for each descriptor on results/[descriptor_name]_res.txt")
    
    io_config.RESULTS_DIR.mkdir(exist_ok=True)
    
    result_files = [(io_config.RESULTS_DIR / f"{name}_res.txt").open("w") for name in descriptors_names]

    for image_num, image_metrics in enumerate(all_metrics):
        for descriptor_type, metric in enumerate(image_metrics):
            
            objective_file = result_files[descriptor_type]
            objective_file.write(f"Image: {image_num:05d}.jpg\n")
            
            for distance_type, distance_name in enumerate(distances_names):
                
                objective_file.write(f"With distance: {distance_name}\n")
                
                objective_file.write(f"Top {K} images:\n")
                distances = np.array(metric[distance_type])
                if isinstance(WANTED_DISTANCES[distance_type], tuple):
                    #We add a very little number to ensure non-zero divisions and stability for small numbers
                    distances = 1 / (distances + 1e-16)
                top_k_res = np.argsort(distances)[:K]
                np.savetxt(objective_file, top_k_res[None], fmt="%d")
                
                objective_file.write(f"Distance values:\n")
                top_k_values = distances[top_k_res]
                np.savetxt(objective_file, top_k_values[None])
                
                objective_file.write(f"Ground truth:\n")
                objective_file.write(f"{ground_truth[image_num][0]}\n")
                
                objective_file.write(f"Ranking of ground truth in evaluation (from 0):\n")
                ranking = np.argsort(np.argsort(distances))
                objective_file.write(f"{ranking[ground_truth[image_num]]}\n")
                
                objective_file.write(f"--------------------------------------------------------------\n")
            
            objective_file.write(f"|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n")
 
def visualize_scores(scores : list, suffix : str = ""):
    
    fig, ax = plt.subplots()
    ax.set_title(f"Scores obtained with {suffix}", fontsize=12, fontweight='bold')
    cax = ax.matshow(scores, cmap="viridis")
    plot_descriptors_names = WANTED_COLOR_DESCRIPTORS_NAMES
    ax.set_xticks(range(len(distances_names)))
    ax.set_yticks(range(len(plot_descriptors_names)))
    ax.set_xticklabels([name.replace("_", " ") for name in distances_names], rotation=90)
    ax.set_yticklabels(plot_descriptors_names)
    
    for i in range(len(plot_descriptors_names)):
        for j in range(len(distances_names)):
            ax.text(j, i, f"{scores[i][j]:.3f}",
                    ha="center", va="center", color="white", fontsize=8)
            
    plt.colorbar(cax)
    plt.savefig(io_config.RESULTS_DIR / f"obtained_scores{suffix}.png", dpi=300, bbox_inches="tight")
     
def resume_results(all_metrics: list, ground_truth: list, eval_ks: list = [io_config.MIN_K , K]):
    """
    Creates a heatmap with distances and descriptors with the AP@K to have a visual approach
    for multiple values of K.
    """

    log.info("Rendering visual results")
    Path("results").mkdir(exist_ok=True)

    for eval_k in eval_ks:
        # A matrix where a_ij is the sum of the scores gotten using descriptor i
        # and distance metric j
        descriptor_scores = [[0 for _ in range(len(distances_names))] for _ in range(len(descriptors_names))]

        for image_num, image_metrics in enumerate(all_metrics):
            for descriptor_type, metric in enumerate(image_metrics):
                for distance_type, distance_name in enumerate(distances_names):
                    distances = np.array(metric[distance_type])
                    if isinstance(WANTED_DISTANCES[distance_type], tuple):
                        distances = 1 / (distances + 1e-16)

                    predictions = np.argsort(distances)[:eval_k]
                    score = metrics.average_precision_k(ground_truth[image_num], predictions, eval_k)
                    descriptor_scores[descriptor_type][distance_type] += score

        for i in range(len(descriptor_scores)):
            for j in range(len(descriptor_scores[0])):
                descriptor_scores[i][j] /= NUMBER_IMAGE_DEV

        visualize_scores(descriptor_scores, suffix=f"_k{eval_k}")
        log.info(f"Rendered results for K={eval_k}")

            

if __name__ == "__main__":
    
    setup_logging()
    log = logging.getLogger(__name__)
    
    file_path = io_config.DEV_DIR / "gt_corresps.pkl"

    with open(file_path, "rb") as f:
        ground_truth = pickle.load(f)
        
    all_descriptors = compute_development_descriptors()
    
    precomputed_descriptors = load_precomputed_descriptors()
    
    all_metrics = compute_distances(all_descriptors, precomputed_descriptors)
    
    write_results(all_metrics, ground_truth)
    
    resume_results(all_metrics, ground_truth)