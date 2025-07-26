# Prototype: AI Model for Calligraphy Evaluation

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

# 1. Stroke Feature Extraction
def extract_stroke_features(character_img: np.ndarray) -> dict:
    gray = cv2.cvtColor(character_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    skeleton = skeletonize(binary // 255)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stroke_lengths = [cv2.arcLength(cnt, False) for cnt in contours]

    return {
        "stroke_count": len(contours),
        "avg_stroke_length": np.mean(stroke_lengths) if stroke_lengths else 0,
        "stroke_variance": np.var(stroke_lengths) if stroke_lengths else 0,
        "skeleton_density": np.sum(skeleton) / skeleton.size
    }

# 2. Character Structure Analysis
def extract_structure_features(character_img: np.ndarray) -> dict:
    gray = cv2.cvtColor(character_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    labeled = label(binary)
    props = regionprops(labeled)

    centroids = [p.centroid for p in props]
    symmetry = abs(np.mean([abs(x - character_img.shape[1] // 2) for x, _ in centroids]))

    return {
        "component_count": len(props),
        "centroid_symmetry": symmetry
    }

# 3. Layout Evaluation
def extract_layout_features(sheet_img: np.ndarray) -> dict:
    gray = cv2.cvtColor(sheet_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    horizontal_projection = np.sum(binary, axis=1)
    vertical_projection = np.sum(binary, axis=0)

    return {
        "horizontal_var": np.var(horizontal_projection),
        "vertical_var": np.var(vertical_projection)
    }

# 4. Combined Evaluation Model
class CalligraphyEvaluator:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100)

    def fit(self, X: List[dict], y: List[float]):
        features = self._dicts_to_vectors(X)
        features = self.scaler.fit_transform(features)
        self.model.fit(features, y)

    def predict(self, X: List[dict]) -> List[float]:
        features = self._dicts_to_vectors(X)
        features = self.scaler.transform(features)
        return self.model.predict(features)

    def _dicts_to_vectors(self, dicts: List[dict]) -> np.ndarray:
        keys = sorted(dicts[0].keys())
        return np.array([[d[k] for k in keys] for d in dicts])

# === Example Usage ===
# character_img = cv2.imread("sample_char.png")
# layout_img = cv2.imread("sample_sheet.png")

# stroke_feat = extract_stroke_features(character_img)
# struct_feat = extract_structure_features(character_img)
# layout_feat = extract_layout_features(layout_img)

# features = {**stroke_feat, **struct_feat, **layout_feat}
# evaluator = CalligraphyEvaluator()
# evaluator.fit([features], [8.5])  # 8.5 is a mock ground truth score
# prediction = evaluator.predict([features])
# print(f"Predicted score: {prediction[0]:.2f}")
