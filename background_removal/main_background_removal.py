import io
import cv2
import numpy as np
from tqdm import tqdm
from config import background_removal_config, io_config
from background_removal.removal_v2 import run_image_v2
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
    mask_1 = run_image_v2(img)
    #mask_2 = get_brackground_mask(img)
    
    ys, xs = np.where(mask_1 > 0)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    cropped = img[y1:y2+1, x1:x2+1]
    

    return cropped, mask_1

def compute_grad_magnitude(img):
    grad_x = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0)
    grad_y = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1)
    grad_module = cv2.magnitude(grad_x, grad_y)
    return np.uint8(grad_module)

def get_discriminative_mask(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv_image)
    edges_s = compute_grad_magnitude(s)

    mask_s = edges_s
    mask_s[mask_s < 75] = 0
    mask_s[mask_s != 0] = 255

    mask = mask_s
    

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(10, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5, 10)))

    # plt.imshow(mask,cmap='gray')
    # plt.show()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    #For each label except background, get area and pick the two largest (possible two paintings)
    component_info = [(i, stats[i, 4]) for i in range(1, num_labels)]
    component_info.sort(key=lambda c: c[1], reverse=True)
    top_two_labels = [c[0] for c in component_info[:2]]

    new_mask = np.zeros_like(mask)

    for label in top_two_labels:
        
        # Create a mask of only this component
        ys, xs = np.where(labels == label)
        points = np.column_stack((xs, ys))
        
        # Get rotated rectangle from these points
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect).astype(int)
        
        # Fill the rectangle in the mask
        cv2.fillPoly(new_mask, [box], 255)
        
    # plt.imshow(new_mask,cmap='gray')
    # plt.show()  
    
    return new_mask

def get_masks(img) -> list:
    
    AREA_THRESHOLD = (0.2*img.shape[0]) * (0.2*img.shape[1])
    discriminative_mask = get_discriminative_mask(img)

    # Check if there is more than one painting (big area)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(discriminative_mask)
    small_areas = np.sum(stats[1:, 4] < AREA_THRESHOLD)
    # print(num_labels - small_areas - 1)
    if num_labels - small_areas - 1 > 1:
        
        #Sort stats from left painting to right painting (in case there are more than 2 in the future)
        stats_no_background = stats[1:]
        stats_no_background = stats_no_background[np.argsort(stats_no_background[:, 0])]
        
        #Compute their distances to get their image slice
        left_painting_x = stats_no_background[0, 0] + stats_no_background[0, 2]
        right_painting_x = stats_no_background[1, 0]
        distance = right_painting_x - left_painting_x
        middle_pos = left_painting_x + distance//2

        
        left_painting = img[:, 0:middle_pos]
        right_painting = img[:, middle_pos:]
        
        cropped_left, left_mask = main_background_removal(left_painting)
        cropped_right, right_mask = main_background_removal(right_painting)
        return [cropped_left, cropped_right], [left_mask, right_mask]

    # print("Doing single mask")
    # plt.imshow(img)
    # plt.show()
    cropped, mask = main_background_removal(img)
    return [cropped], [mask]