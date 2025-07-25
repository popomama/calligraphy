import cv2
import numpy as np
from skimage import filters

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    thresh = filters.threshold_otsu(blur)
    binary = (blur > thresh).astype(np.uint8) * 255
    return binary