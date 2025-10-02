"""
Precomputes the descriptors for all the DDBB images.    
"""

"""

Run from Team3 with:
    python -m descriptors.descriptor_creator
"""

import cv2
import numpy as np
from utils import config
from descriptors.color_descriptors import color_descriptors_func as color_descriptors

# Set up
config.ensure_dirs()

# Process each image
names = [f.__name__ for f in config.WANTED_DESCRIPTORS_OFFLINE]
files = [(config.COLOR_DESC_DIR / f"{name}.txt").open("w") for name in names]


for i in range(config.count_jpgs(config.DB_DIR)):
    image_path = config.db_image_path(i)
    img = cv2.imread(image_path)
    for idx, function in enumerate(config.WANTED_DESCRIPTORS_OFFLINE):
        descriptor = function(img, config.DB_NAME,i,visualize=config.STORE_HISTOGRAMS)
        np.savetxt(files[idx], descriptor[None])
        print(i)
                
for file in files:
    file.close()
                


        
    