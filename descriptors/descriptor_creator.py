"""
Precomputes the descriptors for all the DDBB images.    
"""

"""

To run this without issues because of python not considering it part of the package,
run it from the Team3 folder using python -m descriptors.descriptor_creator

"""

import cv2
import utils.descriptors as descriptors
import numpy as np

NUMBER_OF_FILES = 287
NAME_OF_THE_SET = "BBDD"


wanted_descriptors = [descriptors.gray_descriptor, 
                      descriptors.rgb_descriptor]

#It would be more efficient to pass the loaded image, since we can mantain it in

names = [f.__name__ for f in wanted_descriptors]
files = [open(f"descriptors/{name}.txt", "w") for name in names]

for i in range(NUMBER_OF_FILES):
            image_path = f"data/BBDD/bbdd_{i:05d}.jpg"
            img = cv2.imread(image_path)
            for idx, function in enumerate(wanted_descriptors):
                descriptor = function(img, NAME_OF_THE_SET,i,visualize=True)
                np.savetxt(files[idx], descriptor[None])
                
for file in files:
    file.close()
                


        
    