import io
import cv2
import numpy as np
from tqdm import tqdm
from config import background_removal_config, io_config
from background_removal.ensemble_background_removal_X import run_algorithm_x
from background_removal.ray_background_removal import get_brackground_mask


NUMBER_IMAGE_DEV =io_config.count_jpgs(io_config.DEV_DIR)

def union_mask(mask_1,mask_2):
    return mask_1|mask_2

def intersection_mask(mask_1,mask_2):
    return mask_1 & mask_2


def main_background_removal(img):
    mask_1 = run_algorithm_x(img)
    mask_2 = get_brackground_mask(img)

    return mask_1


if __name__ ==  "__main__":

    precision = 0
    recall = 0
    score = 0

    for i in tqdm(range(NUMBER_IMAGE_DEV), desc="Dev images processed: "):
        image_path = io_config.dev_image_path(i)
        img = cv2.imread(image_path)
        final_mask = main_background_removal(img)

        # Metrics
        image_path = image_path.with_suffix(".png")
        mask_img = cv2.imread(image_path)[:,:,0]

        TP = np.sum(mask_img & final_mask)
        FN = np.sum(mask_img & (255 - final_mask))
        FP = np.sum((255 - mask_img) & final_mask)
        TN = np.sum((255 - mask_img) & (255 - final_mask))

        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F_score = 2*(P*R/(P + R))

        score = score + F_score
        precision += P
        recall += R
        
        print(f"Precision: {P}\nRecall: {R}\nF_score: {F_score}")
    
    print(f"Average F_score : {score/NUMBER_IMAGE_DEV}")
    print(f"Average Precision : {precision/NUMBER_IMAGE_DEV}")
    print(f"Average Recall : {recall/NUMBER_IMAGE_DEV}")
        




