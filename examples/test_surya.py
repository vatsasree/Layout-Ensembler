from PIL import Image
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
import sys

img_path = sys.argv[1]
img = Image.open(img_path)

foundation_pred = FoundationPredictor(device="cuda:0")
layout_pred = LayoutPredictor(foundation_pred)

res = layout_pred([img])[0]
for box in res.bboxes:
    print(box.label, box.bbox)
