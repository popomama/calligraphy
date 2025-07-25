import numpy as np

def score_handwriting(strokes, structure, layout):
    stroke_score = np.mean([s["aspect_ratio"] for s in strokes])  # 粗细均衡
    structure_score = 1.0 - abs(structure["aspect_ratio"] - 0.7)   # 字形比例

    layout_score = (
        0.4 * (1 / (layout["alignment_score"] + 1e-5)) +
        0.3 * layout["density_score"] +
        0.3 * layout["whitespace_score"]
    )

    total_score = 0.5 * stroke_score + 0.3 * structure_score + 0.2 * layout_score
    return round(total_score * 100, 2)