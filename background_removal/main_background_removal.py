import io
import cv2
import numpy as np
from tqdm import tqdm
from config import background_removal_config, io_config
from background_removal.ensemble_background_removal_X import run_image
from background_removal.ray_background_removal import get_brackground_mask
import matplotlib.pyplot as plt


NUMBER_IMAGE_DEV =io_config.count_jpgs(io_config.DEV_DIR)

def make_straight(mask):
    ys, xs = np.where(mask > 0)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    mask[y1:y2, x1:x2] = 255
    return mask

def union_mask(mask_1,mask_2):
    
    res_mask = mask_1|mask_2
    
    res_mask = make_straight(res_mask)
    
    return res_mask

def intersection_mask(mask_1,mask_2):
    
    res_mask = make_straight(mask_1 & mask_2)
    
    return res_mask

def component_mix(mask_1, mask_2):
    
    diff_mask = np.bitwise_xor(mask_1, mask_2)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diff_mask, connectivity=4)
    areas = stats[1:, cv2.CC_STAT_AREA] 
    
    THRESHOLD = (mask_1.shape[0] * mask_1.shape[1]) * 0.1
    
    print(THRESHOLD)
    
    if np.any(areas > THRESHOLD):
        result_mask = intersection_mask(mask_1, mask_2)
    else:
        result_mask = union_mask(mask_1, mask_2)
    
    return result_mask
    

def main_background_removal(img):
    mask_1 = run_image(img)
    #mask_2 = get_brackground_mask(img)
    
    ys, xs = np.where(mask_1 > 0)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    cropped = img[y1:y2+1, x1:x2+1]
    

    return cropped, mask_1

if __name__ ==  "__main__":

    precision = 0
    recall = 0
    score = 0

    for i in tqdm(range(NUMBER_IMAGE_DEV), desc="Dev images processed: "):
        image_path = io_config.dev_image_path(i)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        _, final_mask = main_background_removal(img)

        # Metrics
        image_path = image_path.with_suffix(".png")
        mask_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

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

        print(f"Precision: {P:.4f}\nRecall: {R:.4f}\nF_score: {F_score:.4f}")
    
    print(f"Average F_score : {(score/NUMBER_IMAGE_DEV):.4f}")
    print(f"Average Precision : {(precision/NUMBER_IMAGE_DEV):.4f}")
    print(f"Average Recall : {(recall/NUMBER_IMAGE_DEV):.4f}")
