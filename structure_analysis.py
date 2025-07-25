import cv2
import numpy as np

def analyze_structure(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(cnt)
    cx = int(M["m10"] / (M["m00"] + 1e-5))
    cy = int(M["m01"] / (M["m00"] + 1e-5))
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    return {
        "center": (cx, cy),
        "area": area,
        "bounding_box": (x, y, w, h),
        "aspect_ratio": w / h if h else 0
    }