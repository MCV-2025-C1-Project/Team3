"""
Precomputes the descriptors for all the DDBB images.    
"""

"""

Run from Team3 with:
    python -m descriptors.descriptor_creator
"""

import cv2
import numpy as np
from config import io_config, general_config

ALL_BLOCKS = {}

if "COLOR_DESCRIPTORS" in general_config.DESCRIPTORS:
    from config.color_descriptors_config import WANTED_COLOR_DESCRIPTORS
    ALL_BLOCKS["COLOR_DESCRIPTORS"] = {
        "descriptors": WANTED_COLOR_DESCRIPTORS,
        "dir": io_config.DESCRIPTORS_DIR / "color_descriptors"
    }

# ADD MORE IN THE FUTURE: THE LIST IN GENERAL CONFIG WILL DEFINE THE DESCRIPTORS WE WANT TO CREATE

# Set up
io_config.ensure_dirs()

#Create the directories and paths used to store the descriptors
for block, data in ALL_BLOCKS.items():
    data["dir"].mkdir(parents=True, exist_ok=True)
    names = [f.__name__ for f in data["descriptors"]]
    data["files"] = [(data["dir"] / f"{name}.txt").open("w") for name in names]

#Precompute the descriptors and save them on the predefined file
for i in range(io_config.count_jpgs(io_config.DB_DIR)):
    image_path = io_config.db_image_path(i)
    img = cv2.imread(image_path)

    for block, data in ALL_BLOCKS.items():
        for idx, function in enumerate(data["descriptors"]):
            descriptor = function(img, io_config.DB_NAME, i, visualize=io_config.STORE_HISTOGRAMS)
            np.savetxt(data["files"][idx], descriptor[None])
    print(i)

for block, data in ALL_BLOCKS.items():
    for f in data["files"]:
        f.close()
                


        
    