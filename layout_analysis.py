import numpy as np

def evaluate_layout(binary_img):
    vertical_projection = np.sum(binary_img, axis=0)
    horizontal_projection = np.sum(binary_img, axis=1)
    alignment_score = np.std(horizontal_projection) + np.std(vertical_projection)

    density = np.sum(binary_img) / (binary_img.shape[0] * binary_img.shape[1])
    whitespace_score = 1 - density

    return {
        "alignment_score": alignment_score,
        "density_score": density,
        "whitespace_score": whitespace_score
    }