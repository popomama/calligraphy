import cv2
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    thresh = filters.threshold_otsu(blur)
    binary = (blur > thresh).astype(np.uint8) * 255

    # Visualization
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # axs[0].imshow(img, cmap='gray')
    # axs[0].set_title('Original Grayscale')
    # axs[0].axis('off')

    # axs[1].imshow(blur, cmap='gray')
    # axs[1].set_title('Blurred')
    # axs[1].axis('off')

    # axs[2].imshow(binary, cmap='gray')
    # axs[2].set_title('Binary (Otsu Threshold)')
    # axs[2].axis('off')

    # plt.tight_layout()
    # plt.show()

    return binary

# import cv2
# import matplotlib.pyplot as plt

def extract_strokes(binary_img):
    # Detect edges and contours
    edges = cv2.Canny(binary_img, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare a copy of the image for drawing
    vis_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

    features = []
    for cnt in contours:
        length = cv2.arcLength(cnt, True)
        rect = cv2.boundingRect(cnt)
        width, height = rect[2], rect[3]
        aspect_ratio = width / height if height > 0 else 0

        # Draw the contour and bounding box
        cv2.drawContours(vis_img, [cnt], -1, (0, 255, 0), 2)  # Green contour
        cv2.rectangle(vis_img, (rect[0], rect[1]), (rect[0]+width, rect[1]+height), (0, 0, 255), 2)  # Red box

        features.append({
            "length": length,
            "aspect_ratio": aspect_ratio,
            "bounding_box": rect
        })

    # Visualize results
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Contours and Bounding Boxes")
    plt.axis("off")
    plt.show()

    return features

import cv2
import matplotlib.pyplot as plt

def analyze_structure(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None

    # Grab largest contour
    cnt = max(contours, key=cv2.contourArea)

    # Compute moments and centroid
    M = cv2.moments(cnt)
    cx = int(M["m10"] / (M["m00"] + 1e-5))
    cy = int(M["m01"] / (M["m00"] + 1e-5))

    # Compute area and bounding box
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h else 0

    # Prepare visualization
    vis_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_img, [cnt], -1, (0, 255, 0), 2)              # Draw contour (green)
    cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)     # Draw bounding box (red)
    cv2.circle(vis_img, (cx, cy), 5, (255, 0, 0), -1)                  # Draw centroid (blue)

    # Display using matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Analyzed Structure: Largest Contour, Bounding Box, Centroid")
    plt.axis("off")
    plt.show()

    return {
        "center": (cx, cy),
        "area": area,
        "bounding_box": (x, y, w, h),
        "aspect_ratio": aspect_ratio
    }


import numpy as np
import matplotlib.pyplot as plt

def visualize_layout(binary_img):
    vertical_projection = np.sum(binary_img, axis=0)
    horizontal_projection = np.sum(binary_img, axis=1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Vertical Projection (content per column)
    axs[0].plot(vertical_projection, color='purple')
    axs[0].set_title("📏 Vertical Projection")
    axs[0].set_xlabel("Columns")
    axs[0].set_ylabel("Sum of pixel values")
    axs[0].grid(True)

    # Horizontal Projection (content per row)
    axs[1].plot(horizontal_projection, color='teal')
    axs[1].set_title("📐 Horizontal Projection")
    axs[1].set_xlabel("Rows")
    axs[1].set_ylabel("Sum of pixel values")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    b=10


binary_img = preprocess_image("TestDao.jpg")
#extract_strokes(binary_img)
#nalyze_structure(binary_img)
visualize_layout(binary_img)