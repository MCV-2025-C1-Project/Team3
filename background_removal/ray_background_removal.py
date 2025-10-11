import numpy as np
import cv2
from utils import metrics
from numpy.typing import NDArray

def get_component_hists(img : NDArray):
    """
    Gets normalized histograms of each color component of the image
    
    Parameters
    ----------
        img : NDArray
            An image with 3 color components
            
    Returns
    -------
        out:
            Histograms for each of it's color components
    """
    r_hist, _ = np.histogram(img[:,0], bins=255, range=(0, 255))
    g_hist, _ = np.histogram(img[:,1], bins=255, range=(0, 255))
    b_hist, _ = np.histogram(img[:,2], bins=255, range=(0, 255))
    r_hist_norm = r_hist / (np.sum(r_hist) + 1e-10)
    g_hist_norm = g_hist / (np.sum(g_hist) + 1e-10)
    b_hist_norm = b_hist / (np.sum(b_hist) + 1e-10)
    return (r_hist_norm, g_hist_norm, b_hist_norm)

def line_difference(diff_image : NDArray, start_coords : tuple, direction : tuple, thresh : float) -> tuple:
    """
    Gives the intersection of a ray from the starting coordinates on the direction given with the matrix given
    
    Parameters
    ----------
        diff_image : 2D - NDArray
            An 1 channel image
        starts_coords : tuple (height, width)
            The matrix coordinates from where you send the ray
        direction : tuple (height, width)
            The direction the ray will follow
        thresh : float
            The height the ray travels at
            
    Returns
    -------
        (y, x):
            The coordinates where the ray has impacted
    """
    current_y = start_coords[0]
    current_x = start_coords[1]
    difference = diff_image[current_y, current_x]
    
    while difference < thresh:
        current_y += direction[0]
        current_x += direction[1]
        if current_x == 0 or current_x >= diff_image.shape[1] - 1 or current_y == 0 or current_y >= diff_image.shape[0] - 1:
            return (current_y, current_x)
        difference = np.abs(diff_image[current_y, current_x])
        #print(difference)
    
    return (current_y, current_x)

def line_difference_vertical(img, start_coords : tuple, direction : tuple, thresh : float) -> tuple:
    """
    Gives the position of the first row from the given starting coordinates\n
    and in the given direction that has a major change on it's overall color
    
    Parameters
    ----------
        img : 3D - NDArray
            An image in RGB format
        starts_coords : tuple (height, width)
            The matrix coordinates from where you start moving
        direction : tuple (height, width)
            The direction the ray will follow
        thresh : float
            The max difference considered a major change
            
    Returns
    -------
        (y, x):
            The coordinates where the change is found
    """
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = img_lab.astype('float')
    img_hsv = img_hsv.astype('float')
    current_y = start_coords[0]
    current_x = start_coords[1]
    
    hist_l, hist_a, hist_b = get_component_hists(img_lab[current_y])
    hist_new_l, hist_new_a, hist_new_b = get_component_hists(img_lab[current_y + direction[0]])
    
    hist_h, hist_s, hist_v = get_component_hists(img_hsv[current_y])
    hist_new_h, hist_new_s, hist_new_bv= get_component_hists(img_hsv[current_y + direction[0]])
    
    difference = np.sum([
                  metrics.earth_movers_distance(hist_a, hist_new_a) * 0.45,
                  metrics.earth_movers_distance(hist_b, hist_new_b) * 0.45,
                  metrics.canberra_distance(hist_s, hist_new_s) * 0.1])
    
    while difference < thresh and current_y >= 0 and current_y < img.shape[0]:
        current_y += direction[0]
        hist_l, hist_a, hist_b = get_component_hists(img_lab[current_y])
        hist_new_l, hist_new_a, hist_new_b = get_component_hists(img_lab[current_y + direction[0]])
        
        hist_h, hist_s, hist_v = get_component_hists(img_hsv[current_y])
        hist_new_h, hist_new_s, hist_new_bv= get_component_hists(img_hsv[current_y + direction[0]])
        
        difference = np.sum([
                  metrics.earth_movers_distance(hist_a, hist_new_a) * 0.45,
                  metrics.earth_movers_distance(hist_b, hist_new_b) * 0.45,
                  metrics.canberra_distance(hist_s, hist_new_s) * 0.1])
        #print(difference)
    
    return (current_y, current_x)

def line_difference_horizontal(img, start_coords : tuple, direction : tuple, thresh : float) -> tuple:
    """
    Gives the position of the first column from the given starting coordinates\n
    and in the given direction that has a major change on it's overall color
    
    Parameters
    ----------
        img : 3D - NDArray
            An image in RGB format
        starts_coords : tuple (height, width)
            The matrix coordinates from where you start moving
        direction : tuple (height, width)
            The direction the ray will follow
        thresh : float
            The max difference considered a major change
            
    Returns
    -------
        (y, x):
            The coordinates where the change is found
    """
    
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = img_lab.astype('float')
    img_hsv = img_hsv.astype('float')
    current_y = start_coords[0]
    current_x = start_coords[1]
    
    hist_l, hist_a, hist_b = get_component_hists(img_lab[:, current_x])
    hist_new_l, hist_new_a, hist_new_b = get_component_hists(img_lab[:, current_x + direction[1]])
    
    hist_h, hist_s, hist_v = get_component_hists(img_hsv[:, current_x])
    hist_new_h, hist_new_s, hist_new_bv= get_component_hists(img_hsv[:, current_x + direction[1]])
    
    
    difference = np.sum([
                  metrics.earth_movers_distance(hist_a, hist_new_a) * 0.45,
                  metrics.earth_movers_distance(hist_b, hist_new_b) * 0.45,
                  metrics.canberra_distance(hist_s, hist_new_s) * 0.1])
    
    while difference < thresh:
        current_x += direction[1]
        hist_l, hist_a, hist_b = get_component_hists(img_lab[:, current_x])
        hist_new_l, hist_new_a, hist_new_b = get_component_hists(img_lab[:, current_x + direction[1]])
        
        hist_h, hist_s, hist_v = get_component_hists(img_hsv[:, current_x])
        hist_new_h, hist_new_s, hist_new_bv= get_component_hists(img_hsv[:, current_x + direction[1]])
        
        difference = np.sum([
                  metrics.earth_movers_distance(hist_a, hist_new_a) * 0.45,
                  metrics.earth_movers_distance(hist_b, hist_new_b) * 0.45,
                  metrics.canberra_distance(hist_s, hist_new_s) * 0.1])
        
    
    return (current_y, current_x)

def compute_vertical_line(point1 : tuple, point2 : tuple) -> function:
    """
    Computes the line function between two points considering the row axis as the base axis
    
    Parameters
    ----------
        point1 : tuple
            First point
        point2 : tuple
            Second point
    
    Returns
    -------
        line_func : function
            The function expressed by the line between the two points
    
    """
    A = point1[1] - point2[1]
    B = point2[0] - point1[0]
    C = point1[0]*point2[1] - point2[0]*point1[1]
    
    if B != 0:
        m = -A / B
        b = -C / B
    else:
        m = 0
        b = -C/A
    
    def line_func(x):
        return m*x + b
    
    return line_func

def compute_horizontal_line(point1 : tuple, point2 : tuple) -> function:
    """
    Computes the line function between two points considering the row axis as the base axis
    
    Parameters
    ----------
        point1 : tuple
            First point
        point2 : tuple
            Second point
    
    Returns
    -------
        line_func : function
            The function expressed by the line between the two points
    
    """
    A = point1[0] - point2[0]
    B = point2[1] - point1[1]
    C = point1[1]*point2[0] - point2[1]*point1[0]
    
    if B != 0:
        m = -A / B
        b = -C / B
    else:
        m = 0
        b = -C/A
    
    def line_func(x):
        return m*x + b
    
    return line_func

def get_borders(img : NDArray) -> tuple:
    """
    Computes the borders of the painting
    
    Parameters
    ----------
        img : 3D-NDArray
            An image in RGB format
    
    Returns
    -------
        top_point : tuple
            The point representing the top border
        bottom_point : tuple
            The point representing the bottom border
        left_point : tuple
            The point representing the left border
        right_point : tuple
            The point representing the right border
    
    """
    middle_x = img.shape[1]//2
    middle_y = img.shape[0]//2
    threshold_vert = 5
    

    #We look from the top
    top_point_left = line_difference_vertical(
        img, start_coords=(0, middle_y), direction=(1,0), thresh=threshold_vert)
    
    bottom_point_left = line_difference_vertical(
        img, start_coords=(img.shape[0] - 1, middle_y), direction=(-1, 0), thresh=threshold_vert)
    
    threshold_hor = 4
    
    left_point = line_difference_horizontal(
        img, start_coords=(top_point_left[0], 0), direction=(0, 1), thresh=threshold_hor
    )
    
    right_point = line_difference_horizontal(
        img, start_coords=(top_point_left[0], img.shape[1] - 1), direction=(0, -1), thresh=threshold_hor
    )
        

    return top_point_left, bottom_point_left, left_point, right_point

def compute_vertical_side(img : NDArray, start_coords : tuple, end_coords : tuple, direction : tuple, ray_direction : tuple) -> NDArray:
    """
    Gives the position of all the vertical borders found using the ray approximation
    
    Parameters
    ----------
        img : 3D - NDArray
            An image in RGB format
        starts_coords : tuple (height, width)
            The matrix coordinates from where you start moving
        end_coords : tuple (height, width)
            The matrix coordinates where you stop moving
        direction : tuple (height, width)
            The direction the ray launcher will be moving to
        ray_direction : tuple (height, width)
            The direction the rays will be launched to
            
    Returns
    -------
        points : NDArray
            All the border points found
    """
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype('float')
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL).astype(float)
    diff_image = img_hsv[:, :, 1]
    
    
    def_diff_image = np.abs(diff_image[1:diff_image.shape[0], :] - diff_image[:diff_image.shape[0] - 1, :])
    
    points = []
    current_x = start_coords[1]
    current_y = start_coords[0]
    threshold = 3
    
    while current_x <= end_coords[1] and current_y <= end_coords[0]:
        points.append(line_difference(def_diff_image, start_coords=(current_y, current_x), direction=ray_direction, thresh=threshold))
        current_x += direction[1]
        current_y += direction[0]

    
    return np.array(points)

def compute_bottom_side(img : NDArray, start_coords : tuple, direction : tuple, ray_direction : tuple, threshold : float) -> NDArray:
    """
    Gives the position of all the horizontal borders found using the ray approximation
    
    Parameters
    ----------
        img : 3D - NDArray
            An image in RGB format
        starts_coords : tuple (height, width)
            The matrix coordinates from where you start moving
        direction : tuple (height, width)
            The direction the ray launcher will be moving to
        ray_direction : tuple (height, width)
            The direction the rays will be launched to
        threshold : float
            The height the where the ray is fired from
            
    Returns
    -------
        points : NDArray
            All the border points found
    """
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype('float')
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL).astype(float)
    diff_image = 0.2*img_lab[:,:,1] + 0.2*img_lab[:, :,2] + 0.6*img_hsv[:, :, 1]
    
    def_diff_image = np.abs(diff_image[:, 1:(img.shape[1])] - diff_image[:, :(img.shape[1] - 1)])
    
    """plt.imshow(def_diff_image),
    plt.show()"""
    
    points = []
    current_x = start_coords[1]
    current_y = start_coords[0]
    
    while current_x < def_diff_image.shape[1] and current_y < def_diff_image.shape[0]:
        points.append(line_difference(def_diff_image, start_coords=(current_y, current_x), direction=ray_direction, thresh=threshold))
        current_x += direction[1]
        current_y += direction[0]

    
    return np.array(points)

def get_vertical_func(img : NDArray, top_side : tuple, bottom_side : tuple, ray_direction : tuple, compare_median : int, shift_y : int) -> function:
    """
    Gives the function defined by a vertical border near a given median using a ray approximation
    
    Parameters
    ----------
        img : 3D - NDArray
            An image in RGB format
        top_side : tuple (height, width)
            The matrix coordinates from where you seek the top point of the line
        bottom_side : tuple (height, width)
            The matrix coordinates from where you seek the bottom point of the line
        ray_direction : tuple (height, width)
            The direction the rays will be launched to
        compare_median : int
            The coordinate to compare to when finding the points
        shift_y : int
            The radius of surrounding pixels surveyed to get the bottom and top points
            
    Returns
    -------
        line_func : function
            The function that characterizes the vertical border near the given `compare_median`
    """
    top_side = compute_vertical_side(img, (top_side[0], top_side[1]), (top_side[0]+shift_y, top_side[1]), (1, 0), ray_direction)
    correct_top = top_side[(top_side[:,1] >= compare_median - 20) & (top_side[:, 1] <= compare_median + 20)]
    top_median = int(np.median(correct_top[:, 1])) if len(correct_top) > 0 else compare_median

    bottom_side = compute_vertical_side(img, (bottom_side[0] - shift_y, bottom_side[1]), (bottom_side[0], bottom_side[1]), (1, 0), ray_direction)
    correct_bottom = bottom_side[(bottom_side[:,1] >= compare_median - 20) & (bottom_side[:, 1] <= compare_median + 20)]
    bottom_median = int(np.median(correct_bottom[:, 1])) if len(correct_bottom) > 0 else compare_median
    
    try:
        candidates = np.where(correct_bottom[: , 1] == bottom_median)
        max_side = (np.argmax(correct_bottom[candidates[0]][:, 0]))

        bottom_point = correct_bottom[max_side]

        candidates = np.where(correct_top[: , 1] == top_median)
        max_side = (np.argmax(correct_top[candidates[0]][:, 0]))

        top_point = correct_top[max_side]

        line_func = compute_vertical_line(top_point, bottom_point)
    except:
        line_func = compute_vertical_line((0, compare_median), (10, compare_median))

    return line_func

def get_brackground_mask(img : NDArray) -> NDArray:
    """
    It computs the mask to remove the background of the painting
    
    Parameters
    ----------
        img : 3D-NDArray
            An BGR image
    
    Returns
    -------
        mask : 2D-NDArray
            A 1 channel image of the same size of `img` containing the background mask
    """
    middle_y = img.shape[0] // 2
    shift_y = middle_y // 4

    # Get points by whole row/column aware methods
    top_point, bottom_point, left_point, right_point = get_borders(img)

    #Combine row aware with ray throw to get a corrected bottom side
    bottom_side = compute_bottom_side(img, (img.shape[0] - 1, 0), (0, 1), (-1, 0), threshold=4)
    bottom_median= int(np.median(bottom_side[:, 0]))

    correct_values = bottom_side[(bottom_side[:,0] <= bottom_point[0] +10) & (bottom_side[:, 0] >= bottom_median - 10)]
    corrected_median = int(np.median(correct_values[:, 0])) if len(correct_values) > 0 else bottom_point[0]
    
    top_side = compute_bottom_side(img, (0, 0), (0, 1), (1, 0), threshold=2)
    top_median= int(np.median(top_side[:, 0]))

    correct_top_values = top_side[(top_side[:,0] >= top_point[0] + 10) & (top_side[:, 0] <= top_median - 10)]
    corrected_top_median = int(np.median(correct_top_values[:, 0])) if len(correct_top_values) > 0 else top_point[0]

    border_image = img.copy()

    left_side = compute_vertical_side(img, (top_point[0], 0), (corrected_median, 0), (1, 0), (0, 1))
    left_median = int(np.median(left_side[:, 1]))
    correct_left_values = left_side[(left_side[:,1] >= left_point[1] - 10) & (left_side[:, 1] <= left_median - 10)]
    corrected_left_median = int(np.median(correct_left_values[:, 1])) if len(correct_left_values) > 0 else left_point[1]

    right_side = compute_vertical_side(img, (top_point[0], img.shape[1] - 1), (corrected_median, img.shape[1] - 1), (1, 0), (0, -1))
    right_median = int(np.median(right_side[:, 1]))
    correct_right_values = right_side[(right_side[:,1] <= right_point[1] +10) & (right_side[:, 1] >= right_median - 10)]
    corrected_right_median = int(np.median(correct_right_values[:, 1])) if len(correct_right_values) > 0 else right_point[1]
    
    left_line_func = get_vertical_func(img, (top_point[0], 0), (corrected_median, 0), (0, 1), corrected_left_median, shift_y)
    right_line_func = get_vertical_func(img, (top_point[0], img.shape[1] - 1), (corrected_median, img.shape[1] - 1), (0, -1), corrected_right_median, shift_y)

    y_left = np.array([left_line_func(x) for x in range(img.shape[0])])

    W = img.shape[1]
    cols = np.arange(W)
    mask_left = cols < y_left[:, None]

    y_right = np.array([right_line_func(x) for x in range(img.shape[0])])

    W = img.shape[1]
    mask_right = cols > y_right[:, None]

    mask = mask_left | mask_right

    border_image[mask] = 0
    mask_middle = (cols >= y_left[:, None]) & (cols <= y_right[:, None])
    border_image[mask_middle] = 255
    border_image[range(img.shape[0] - 1, corrected_median, -1), :, :] = 0
    border_image[range(corrected_top_median), :, :] = 0
    
    result = border_image[:, :, 0]
    
    return result