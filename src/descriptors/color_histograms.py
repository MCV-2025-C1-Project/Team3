import cv2
import matplotlib.pyplot as plt


def compute_color_histogram(image_path: str, save_img: bool = False) -> list:
    """
    Input:
    -image_path (str): Path to the image
    
    Returns:
    -hist: BGR histogram concatenated by colors
    """
    
    img = cv2.imread(image_path)
    img = img.astype(int)
 
    hist = [0] * (256*3)

    for row in img:
        for col in row:
            hist[col[0]] += 1
            hist[col[1] + 256] += 1
            hist[col[2] + 512] += 1

    if save_img:    
        plt.figure(figsize=(12, 4))
        plt.bar(range(768), hist, width=1.0, color="black")
        plt.xlabel("Concatenated BGR bins (0-255 B, 256-511 G, 512-767 R)")
        plt.ylabel("Frequency")
        plt.title("Concatenated BGR Histogram")
        plt.savefig("histogram.png", dpi=300, bbox_inches="tight")

    return hist
    


compute_color_histogram("./BBDD/bbdd_00000.jpg", True)