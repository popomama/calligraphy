import numpy as np

def score_handwriting1(strokes, structure, layout):
    stroke_score = np.mean([s["aspect_ratio"] for s in strokes])  # 粗细均衡
    structure_score = 1.0 - abs(structure["aspect_ratio"] - 0.7)   # 字形比例

    # layout_score = (
    #     0.4 * (1 / (layout["alignment_score"] + 1e-5)) +
    #     0.3 * layout["density_score"] +
    #     0.3 * layout["whitespace_score"]
    # )

    layout_score = (
        0.35 * layout["alignment_score"] +       # 行列对齐度
        0.25 * layout["density_score"] +         # 墨迹密度
        0.2  * layout["whitespace_score"] +      # 留白比例
        0.2  * layout["stability_score"]         # 重心稳定性
    )


    total_score = 0.5 * stroke_score + 0.3 * structure_score + 0.2 * layout_score
    return round(total_score * 100, 2)


# def score_handwriting(strokes, structure, layout, weights=None):
#     """
#     综合评估手写内容的质量，基于抽象维度：笔画、结构、布局。

#     输入：
#         strokes: dict，笔画相关评分项（例如：smoothness、count、continuity）
#         structure: dict，字形结构相关评分项（例如：aspect_ratio、symmetry）
#         layout: dict，整体布局相关评分项（例如：whitespace_ratio、alignment）
#         weights: dict，可选参数，设置各维度权重（默认均等）

#     输出：
#         dict，包含每一项维度得分和最终总分
#     """
#     def average_score(score_dict):
#         valid_scores = [v for v in score_dict.values() if isinstance(v, (int, float))]
#         return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

#     # 默认权重设定（可根据训练或专家反馈调整）
#     if weights is None:
#         weights = {
#             "strokes": 1.0,
#             "structure": 1.0,
#             "layout": 1.0
#         }

#     # 分维度评分
#     stroke_score = average_score(strokes)
#     structure_score = average_score(structure)
#     layout_score = average_score(layout)

#     # 综合评分（加权平均）
#     total_weight = sum(weights.values())
#     final_score = (
#         stroke_score * weights["strokes"] +
#         structure_score * weights["structure"] +
#         layout_score * weights["layout"]
#     ) / total_weight

#     # 输出
#     return {
#         "stroke_score": round(stroke_score, 4),
#         "structure_score": round(structure_score, 4),
#         "layout_score": round(layout_score, 4),
#         "final_score": round(final_score, 4)
#     }

def score_aspect_ratio(ratio, ideal=0.7):
    return 1.0 - abs(ratio - ideal)  # 得分越接近 ideal 越高


# def normalize(score, min_val=0, max_val=1.0):
#     return (score - min_val) / (max_val - min_val + 1e-5)

def normalize(score, min_val, max_val):
    """将评分压缩到 0–1 范围，避免异常值主导整体评分"""
    return np.clip((score - min_val) / (max_val - min_val), 0, 1)

def compute_layout_score(raw_scores, weights=None, bounds=None):
    # 默认等权重
    if weights is None:
        weights = {
            "alignment_score": 0.25,
            "density_score": 0.25,
            "stability_score": 0.25,
            "whitespace_score": 0.25
        }

    # 默认归一化区间，根据经验设定
    if bounds is None:
        bounds = {
            "alignment_score": (0.5, 5.0),  # 标准差倒数，一般波动在这范围
            "density_score": (0.0, 0.6),    # 墨迹密度，黑字占比
            "stability_score": (0.0, 1.0),  # 居中性
            "whitespace_score": (0.3, 1.0)  # 留白比例
        }

    # 归一化每个维度
    normalized_scores = {
        key: normalize(raw_scores[key], *bounds[key]) for key in raw_scores
    }

    # 加权求总分
    layout_score = sum(normalized_scores[key] * weights.get(key, 0) for key in normalized_scores)
    return layout_score


def score_handwriting(strokes, structure, layout):
    """
    综合评估手写内容的质量，融合 stroke/structure/layout 三个维度。

    输入：
        strokes: List[dict]，每个笔画包含分析后的特征，比如 aspect_ratio
        structure: dict，整体字形结构特征
        layout: dict，图像层面排版与密度特征

    输出：
        dict，包含分维度评分与总分
    """

    # 笔画评分：以 aspect_ratio 衡量粗细均衡性
    #stroke_score = np.mean([s["aspect_ratio"] for s in strokes])
    stroke_score = np.mean([score_aspect_ratio(s["aspect_ratio"]) for s in strokes])
    # 结构评分：理想 aspect_ratio 偏向 0.7（经验值）
    structure_score = 1.0 - abs(structure.get("aspect_ratio", 0.7) - 0.7)

    # 布局评分：融合多个视觉几何指标（可调权重）
    # layout_score = (
    #     #0.35 * layout.get("alignment_score", 0.0) +
    #     0.35 * normalize(layout["alignment_score"], 0.0, 5.0) +
    #     0.25 * layout.get("density_score", 0.0) +
    #     0.2  * layout.get("whitespace_score", 0.0) +
    #     0.2  * layout.get("stability_score", 0.0)
    # )

    layout_score=  compute_layout_score(layout)


    # 综合评分：目前为等权加权平均
    final_score = (stroke_score + structure_score + layout_score) / 3.0

    return {
        "stroke_score": round(stroke_score, 4),
        "structure_score": round(structure_score, 4),
        "layout_score": round(layout_score, 4),
        "final_score": round(final_score, 4)
    }