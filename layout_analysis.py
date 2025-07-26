# import numpy as np

# def evaluate_layout(binary_img):
#     vertical_projection = np.sum(binary_img, axis=0)
#     horizontal_projection = np.sum(binary_img, axis=1)
#     alignment_score = np.std(horizontal_projection) + np.std(vertical_projection)

#     density = np.sum(binary_img) / (binary_img.shape[0] * binary_img.shape[1])
#     whitespace_score = 1 - density

#     return {
#         "alignment_score": alignment_score,
#         "density_score": density,
#         "whitespace_score": whitespace_score
#     }

import numpy as np
import cv2

def evaluate_layout(binary_img):
    height, width = binary_img.shape

    # 1️⃣ 垂直 & 水平投影
    vertical_projection = np.sum(binary_img, axis=0)
    horizontal_projection = np.sum(binary_img, axis=1)

    alignment_score = (
        1.0 / (np.std(vertical_projection) + 1e-5) + 
        1.0 / (np.std(horizontal_projection) + 1e-5)
    )  # 越平稳越好

    # 2️⃣ 墨迹密度图：衡量整体布局密集度
    density_score = np.sum(binary_img) / (height * width)

    # 3️⃣ 重心稳定性：计算墨迹中心与图像中心偏移
    coords = np.column_stack(np.where(binary_img > 0))
    if coords.size == 0:
        center_shift = 0
    else:
        ink_center = np.mean(coords, axis=0)
        image_center = np.array([height / 2, width / 2])
        shift_vector = ink_center - image_center
        center_shift = np.linalg.norm(shift_vector) / np.sqrt(height**2 + width**2)

    stability_score = 1.0 - center_shift  # 偏移越小越稳定

    # 4️⃣ 空间节奏感（留白比例）
    whitespace_score = 1.0 - density_score

    return {
        "alignment_score": alignment_score,
        "density_score": density_score,
        "stability_score": stability_score,
        "whitespace_score": whitespace_score
    }