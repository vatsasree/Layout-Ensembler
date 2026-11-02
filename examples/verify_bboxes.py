import json
import re
import cv2
import numpy as np
from pathlib import Path

def get_color(label):
    # Hash the label to generate a consistent color
    h = hash(label)
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = h & 0x0000FF
    
    # Mix with white to make colors pastel/bright and visible
    r = (r + 255) // 2
    g = (g + 255) // 2
    b = (b + 255) // 2
    return (b, g, r) # OpenCV uses BGR

def parse_localize_data(js_path):
    with open(js_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by LOCALIZE_DATA[
    parts = content.split('LOCALIZE_DATA[')
    
    data = []
    for part in parts:
        if not part.strip():
            continue
        
        # Extract the JSON string
        # Format is roughly: '"name"] = { ... }'
        try:
            # Find the first '='
            eq_idx = part.find('=')
            if eq_idx != -1:
                json_str = part[eq_idx+1:].strip()
                # Parse JSON
                parsed = json.loads(json_str)
                data.append(parsed)
        except json.JSONDecodeError as e:
            print(f"Error parsing a block: {e}")
            pass
            
    return data

def main():
    js_path = Path("localize_data.js")
    if not js_path.exists():
        print("localize_data.js not found.")
        return

    data = parse_localize_data(js_path)
    print(f"Parsed {len(data)} pages from {js_path}")

    # Map labels to colors so they are consistent across images
    unique_labels = set()
    for page_data in data:
        for crop in page_data.get("crops", []):
            unique_labels.add(crop["label"])
            
    # Hardcode some nice colors for common labels, fallback to hash
    color_map = {
        "Picture": (255, 200, 200),
        "Title": (200, 200, 255),
        "Section-title": (200, 255, 200),
        "Text": (255, 255, 200),
        "Paragraph": (255, 255, 200),
        "Table": (200, 255, 255),
        "List": (255, 200, 255),
        "Header": (150, 150, 200),
        "Footer": (150, 200, 150)
    }
    
    for lbl in unique_labels:
        if lbl not in color_map:
            color_map[lbl] = get_color(lbl)

    for page_data in data:
        page_name = page_data["page"]
        img_path = Path(f"{page_name}.png")
        if not img_path.exists():
            print(f"Image {img_path} not found. Skipping...")
            continue

        print(f"Processing {img_path}...")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load {img_path}")
            continue

        overlay = img.copy()
        h, w = img.shape[:2]

        for crop in page_data.get("crops", []):
            label = crop["label"]
            bbox = crop["bbox"] # [frac_x, frac_y, frac_w, frac_h]
            
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            bw = int(bbox[2] * w)
            bh = int(bbox[3] * h)
            x2 = x1 + bw
            y2 = y1 + bh
            
            color = color_map.get(label, (128, 128, 128))
            
            # Draw filled rectangle on overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
        # Blend overlay with original image (60% transparent -> alpha = 0.4 for overlay, 0.6 for original)
        # Wait, 60% transparent opacity means opacity is 40% (alpha=0.4)? Or alpha=0.6?
        # Let's use 0.6 for overlay, 0.4 for original to make it visible, or 0.4 overlay 0.6 original.
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Now draw the borders and text so they are solid (not transparent)
        for crop in page_data.get("crops", []):
            label = crop["label"]
            bbox = crop["bbox"]
            
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            bw = int(bbox[2] * w)
            bh = int(bbox[3] * h)
            x2 = x1 + bw
            y2 = y1 + bh
            
            color = color_map.get(label, (128, 128, 128))
            
            # Draw solid border
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label text background
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            # Draw label text
            # cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out_path = f"{page_name}_viz.png"
        cv2.imwrite(out_path, img)
        print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
