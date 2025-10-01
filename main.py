from utils import descriptors, metrics
import cv2
import logging
import logging.config

def setup_logging():
    """Setup logging configuration from .ini file"""
    logging.config.fileConfig("utils/logging.ini", disable_existing_loggers=False)


if __name__ == "__main__":
    
    setup_logging()
    log = logging.getLogger(__name__)
    log.info("Info example message")
    log.debug("Debug example message")
    
    img = cv2.imread("data/BBDD/bbdd_00000.jpg", cv2.IMREAD_GRAYSCALE)
    gray_hist = descriptors.compute_histogram(img)
    print(metrics.euclidean_distance(img, img))
    