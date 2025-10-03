import argparse
import config
from utils import metrics
from config.color_descriptors_config import PREDICTING_COLOR_DESCRIPTORS
from config import general_config, io_config
from pathlib import Path
import cv2
import numpy as np

descriptors_names = [f[0].__name__ for f in PREDICTING_COLOR_DESCRIPTORS]
distances_names = [
    d[0].__name__ if isinstance(d, tuple) else d.__name__
    for d in PREDICTING_COLOR_DESCRIPTORS
]

files = []
for name in descriptors_names:
    files.append((io_config.COLOR_DESC_DIR / f"{name}.txt").open("r"))

def load_precomputed_descriptors() -> list:
    """
    Loads the precomputated descriptors from the BBDD dataset
    """
    
    precomputed_descriptors = []
    
    #Reading of precomputed descriptors on the BBDD
    for file in files:
        images_descriptors = []
        for line in file:
            arr = np.fromstring(line, sep=" ")
            images_descriptors.append(arr)
        file.close()
        
        precomputed_descriptors.append(images_descriptors)

    return precomputed_descriptors


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Computes the most similar museum painting to the images on the given folder",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("folder", help="Folder where the images to predict are located")
    parser.add_argument("-k", type=int, default=5, help="The number of outputs to compute AP@K")
    # The methods are defined on the color descriptors config
    parser.add_argument("--method", type=int, default=1, choices=[1, 2], help="Method to use (default = 1):\n"
                                                    "   1 - HSV histograms with Canberra distance\n"
                                                    "   2 - LAB with hellinger kernel\n"
                                                    )
    

    args = parser.parse_args()

    used_descriptor, used_distance = PREDICTING_COLOR_DESCRIPTORS[args.method - 1]

    images = [p for p in Path(args.folder).iterdir() if p.suffix.lower() == ".jpg"]

    predictions = []
    res_distances = []
    
    precomputed_descriptors = load_precomputed_descriptors()

    images = sorted(images)

    for image in images:
        img = cv2.imread(image)
        image_descriptors = used_descriptor(img)
        distances = []
        for objective_descriptor in precomputed_descriptors[args.method - 1]:
            distance = used_distance(objective_descriptor, image_descriptors)
            if args.method == 2:
                distance = 1 / (distance + 1e-16)
            distances.append(distance)
        distances = np.array(distances)
        predicted = np.argsort(distances)[:args.k].tolist()
        predictions.append(predicted)
        res_distances.append(distances[predicted])

    print(predictions)


    
