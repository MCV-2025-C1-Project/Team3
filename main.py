from utils import descriptors, metrics
import cv2
import logging
import logging.config
import pickle
import numpy as np

DESCRIPTOR_FILE = "gray_example.txt"

def setup_logging():
    """Setup logging configuration from .ini file"""
    logging.config.fileConfig("utils/logging.ini", disable_existing_loggers=False)


if __name__ == "__main__":
    
    setup_logging()
    log = logging.getLogger(__name__)
    
    file_path = "data/qsd1_w1/gt_corresps.pkl"

    with open(file_path, "rb") as f:
        ground_truth = pickle.load(f)
    
    image_path = f"data/qsd1_w1/{0:05d}.jpg"
    log.debug(f"Using image_path: {image_path}")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    descriptor = descriptors.compute_histogram(img)
    
    #Load all the precomputed descriptors
    gray_descriptors = []
    with open(f"descriptors/{DESCRIPTOR_FILE}", "r") as f:
        for line in f:
            arr = np.fromstring(line, sep=" ")
            gray_descriptors.append(arr)
    
    log.info("Precomputed descriptors loaded")
    
    final_metrics = np.array([0]*len(gray_descriptors), dtype=np.float32)
    
    for idx, i in enumerate(gray_descriptors):
        final_metrics[idx] = metrics.euclidean_distance(descriptor, i)
        
    log.info("Computed metrics against all database")
    
    #Prints top 5 and their metric values
    best_distances = np.argsort(final_metrics)
    ranks = np.argsort(best_distances)
    print("Best found candidates:")
    print(best_distances[:5])
    print(final_metrics[best_distances[:5]])
    
    #Prints the expected result for the selected image and where it is 
    print("Expected image:")
    print(ground_truth[0])
    print("Ranking position of the expected image:")
    print(f"{ranks[ground_truth[0]][0]} position")
        