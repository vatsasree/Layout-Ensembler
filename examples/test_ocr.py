from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.layout import LayoutPredictor
from surya.foundation import FoundationPredictor
from surya.settings import settings
import sys

img_path = sys.argv[1]
img = Image.open(img_path).convert("RGB")

print("Loading detectors...")
foundation_pred = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
layout_pred = LayoutPredictor(foundation_pred)

det_predictor = DetectionPredictor()
rec_foundation_pred = FoundationPredictor()
rec_predictor = RecognitionPredictor(rec_foundation_pred)

print("Running Layout Detection...")
layout_results = layout_pred([img])[0]
layout_boxes = layout_results.bboxes

print("Running Text Detection...")
det_results = det_predictor([img])[0]
line_polys = [b.polygon for b in det_results.bboxes]

print("Running Text Recognition...")
rec_results = rec_predictor([img], [line_polys])[0]

# Now let's map them
def get_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]

print("Mapping...")
for lbox in layout_boxes:
    l_poly = lbox.polygon
    lx1, ly1, lx2, ly2 = get_bbox(l_poly)
    
    texts = []
    for line_poly, rec_line in zip(line_polys, rec_results.text_lines):
        cx = sum([p[0] for p in line_poly]) / 4
        cy = sum([p[1] for p in line_poly]) / 4
        
        if lx1 <= cx <= lx2 and ly1 <= cy <= ly2:
            texts.append(rec_line.text)
            
    print(f"Layout [{lbox.label}]: {' '.join(texts)}")
