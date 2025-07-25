from preprocess import preprocess_image
from stroke_features import extract_strokes
from structure_analysis import analyze_structure
from layout_analysis import evaluate_layout
from scoring import score_handwriting

def evaluate(image_path):
    img = preprocess_image(image_path)
    strokes = extract_strokes(img)
    structure = analyze_structure(img)
    layout = evaluate_layout(img)
    score = score_handwriting(strokes, structure, layout)
    print("硬笔书法综合评分:", score)

if __name__ == "__main__":
    evaluate("TestDao.jpg")