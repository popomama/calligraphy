import cv2

def extract_strokes(binary_img):
    edges = cv2.Canny(binary_img, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for cnt in contours:
        length = cv2.arcLength(cnt, True)
        rect = cv2.boundingRect(cnt)
        width, height = rect[2], rect[3]
        aspect_ratio = width / height if height > 0 else 0
        features.append({
            "length": length,
            "aspect_ratio": aspect_ratio,
            "bounding_box": rect
        })
    return features