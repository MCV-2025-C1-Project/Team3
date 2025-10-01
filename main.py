from utils import descriptors, metrics
import cv2
import logging
import logging.config
import pickle
import numpy as np
from pathlib import Path

logging.getLogger("PIL").setLevel(logging.WARNING)

NAME_OF_THE_DEV_SET = "qsd1_w1"

# These are the considered descriptors
wanted_descriptors = [descriptors.gray_descriptor, 
                      descriptors.rgb_descriptor
                      ]

wanted_distances   = [metrics.euclidean_distance,
                      metrics.x2_dist
                      ]

descriptors_names = [f.__name__ for f in wanted_descriptors]
distances_names   = [f.__name__ for f in wanted_distances]

#The amout of results showed (top k)
K = 5

#The precomputed files for the BBDD images must exist
files = [open(f"descriptors/{name}.txt", "r") for name in descriptors_names]

def setup_logging():
    """Setup logging configuration from .ini file"""
    logging.config.fileConfig("utils/logging.ini", disable_existing_loggers=False)
    
def compute_development_descriptors() -> list:
    """Computes the descriptors of the development set"""
    all_descriptors = []
    
    log.info("Computing all the descriptor for development images")
    
    #For each image in the development folder, computes the specified descriptors
    for i in range(30):
        image_path = f"data/{NAME_OF_THE_DEV_SET}/{i:05d}.jpg"
        img = cv2.imread(image_path)
        image_descriptors = []
        for function in wanted_descriptors:
            descriptor = function(img, NAME_OF_THE_DEV_SET, i, visualize=False)
            image_descriptors.append(descriptor)
        all_descriptors.append(image_descriptors)
        
    
    log.info("Descriptors for development images computed")
    
    return all_descriptors

def load_precomputed_descriptors() -> list:
    """Loads the precomputated descriptors from the BBDD dataset"""
    log.info("Loading precomputed descriptors of BBDD images")
    
    precomputed_descriptors = []
    
    i = 0
    
    #Reading of precomputed descriptors on the BBDD
    for file in files:
        images_descriptors = []
        for line in file:
            i += 1
            arr = np.fromstring(line, sep=" ")
            images_descriptors.append(arr)
        file.close()
        
        precomputed_descriptors.append(images_descriptors)
        
        
    log.info("Loaded")

    return precomputed_descriptors

def compute_distances(all_descriptors : list, precomputed_descriptors : list) -> list:
    """Computes the distances between the development and BBDD datasets"""
    log.info("Computing distances between each development and BBDD image")
    
    all_metrics = []
    
    #Computation of the euclidian distance between development and BBDD images
    
    for objective_image in all_descriptors:
        image_metrics = []
        for idx, descriptor in enumerate(objective_image):
            found_metrics = []
            objective_descriptors = precomputed_descriptors[idx]
            for distance_function in wanted_distances:
                distances = []
                for objective_descriptor in objective_descriptors:
                #This distance can be changed as desired. Later on I can make also that
                #all distance metrics are computed, but is probably irrelevant.
                    distances.append(distance_function(descriptor, objective_descriptor))
                found_metrics.append(distances)
            image_metrics.append(found_metrics)
            
        all_metrics.append(image_metrics)
        

    log.info("Computed")
    
    return all_metrics

def write_results(all_metrics, ground_truth : list) -> list:
    """Writes the results on a txt. One per descriptor used"""
    log.info("Outputting results for each descriptor on results/[descriptor_name]_res.txt")
    
    Path("results").mkdir(exist_ok=True)
    
    result_files = [open(f"results/{name}_res.txt", "w") for name in descriptors_names]
    
    for image_num, image_metrics in enumerate(all_metrics):
        for descriptor_type, metric in enumerate(image_metrics):
            
            objective_file = result_files[descriptor_type]
            objective_file.write(f"Image: {image_num:05d}.jpg\n")
            
            for distance_type, distance_name in enumerate(distances_names):
                
                objective_file.write(f"With distance: {distance_name}\n")
                
                objective_file.write(f"Top {K} images:\n")
                distances = np.array(metric[distance_type])
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
    

if __name__ == "__main__":
    
    setup_logging()
    log = logging.getLogger(__name__)
    
    file_path = f"data/{NAME_OF_THE_DEV_SET}/gt_corresps.pkl"


    with open(file_path, "rb") as f:
        ground_truth = pickle.load(f)
        
    all_descriptors = compute_development_descriptors()
    
    precomputed_descriptors = load_precomputed_descriptors()
    
    all_metrics = compute_distances(all_descriptors, precomputed_descriptors)
    
    write_results(all_metrics, ground_truth)