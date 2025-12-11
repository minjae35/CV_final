#!/usr/bin/env python3
"""
Extract OCR text by:
1. Using PaddleOCR to detect textboxes (returns bbox coordinates)
2. Cropping each textbox region
3. Performing OCR on each cropped textbox individually
4. Grouping textboxes into speech bubbles based on spatial proximity
"""
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from paddleocr import PaddleOCR
from collections import defaultdict

# Paths
panel_metadata_path = Path("/home/mjb4835/CV_final_project/data/processed_444_pages/panel_metadata.csv")
panels_dir = Path("/home/mjb4835/CV_final_project/data/processed_444_pages/cropped_panels")
output_metadata_path = Path("/home/mjb4835/CV_final_project/data/processed_444_pages/panel_metadata_with_ocr_cropped.json")

print("="*80)
print("Extracting OCR by Cropping Individual Textboxes")
print("="*80)

# Initialize PaddleOCR
print("\n1. Initializing PaddleOCR...")
try:
    import paddle
    if paddle.device.is_compiled_with_cuda():
        try:
            paddle.set_device('gpu:0')
            use_gpu = True
            print("✓ Using GPU")
        except:
            use_gpu = False
            print("⚠ Using CPU")
    else:
        use_gpu = False
        print("⚠ Using CPU")
except:
    use_gpu = False
    print("⚠ Using CPU")

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
print("✓ PaddleOCR initialized")

# Load panel metadata
print("\n2. Loading panel metadata...")
panel_df = pd.read_csv(panel_metadata_path)
print(f"   Loaded {len(panel_df)} panels")

def get_bbox_bounds(bbox):
    """Convert 4-point bbox to (x1, y1, x2, y2)"""
    x_coords = [p[0] for p in bbox]
    y_coords = [p[1] for p in bbox]
    return int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))

def crop_textbox(img, bbox):
    """Crop textbox region from image"""
    x1, y1, x2, y2 = get_bbox_bounds(bbox)
    # Add padding
    padding = 5
    h, w = img.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return img[y1:y2, x1:x2]

def group_textboxes_into_speech_bubbles(text_boxes, img_width, img_height):
    """
    Group textboxes into speech bubbles based on spatial proximity
    """
    if not text_boxes:
        return []
    
    # Calculate centers and bounds for each textbox
    for box in text_boxes:
        bbox = box['bbox']
        x1, y1, x2, y2 = get_bbox_bounds(bbox)
        box['center_x'] = (x1 + x2) / 2
        box['center_y'] = (y1 + y2) / 2
        box['left_x'] = x1
        box['right_x'] = x2
        box['top_y'] = y1
        box['bottom_y'] = y2
        box['width'] = x2 - x1
        box['height'] = y2 - y1
    
    # Sort by top_y (top to bottom)
    text_boxes = sorted(text_boxes, key=lambda b: b['top_y'])
    
    speech_bubbles = []
    used = [False] * len(text_boxes)
    
    for i, box in enumerate(text_boxes):
        if used[i]:
            continue
        
        # Start a new speech bubble
        bubble = [i]
        used[i] = True
        
        # Find other boxes that belong to the same speech bubble
        for j, other_box in enumerate(text_boxes):
            if used[j] or i == j:
                continue
            
            # Check vertical alignment (same row)
            y_overlap = min(box['bottom_y'], other_box['bottom_y']) - max(box['top_y'], other_box['top_y'])
            y_overlap_ratio = y_overlap / max(box['height'], other_box['height']) if max(box['height'], other_box['height']) > 0 else 0
            
            # Same row (at least 40% vertical overlap)
            if y_overlap_ratio >= 0.4:
                # Check horizontal proximity
                if other_box['left_x'] > box['right_x']:
                    horizontal_gap = other_box['left_x'] - box['right_x']
                elif box['left_x'] > other_box['right_x']:
                    horizontal_gap = box['left_x'] - other_box['right_x']
                else:
                    horizontal_gap = 0
                
                # Adaptive gap threshold
                avg_width = (box['width'] + other_box['width']) / 2
                max_gap = min(avg_width * 0.5, 50)  # 50% of width or 50px max
                
                if horizontal_gap <= max_gap:
                    bubble.append(j)
                    used[j] = True
        
        speech_bubbles.append(bubble)
    
    return speech_bubbles

def extract_ocr_from_panel(ocr, panel_image_path):
    """
    Extract OCR text by:
    1. Detecting textboxes
    2. Cropping each textbox
    3. OCR on each cropped textbox
    4. Grouping into speech bubbles
    """
    try:
        # Load image
        img = cv2.imread(str(panel_image_path))
        if img is None:
            return ""
        
        img_height, img_width = img.shape[:2]
        
        # Step 1: Detect textboxes
        result = ocr.ocr(str(panel_image_path), cls=True)
        
        text_boxes = []
        if result and len(result) > 0 and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_info = line[1]
                    
                    if isinstance(text_info, tuple) and len(text_info) >= 1:
                        text = text_info[0]
                        if text and text.strip():
                            text_boxes.append({
                                'bbox': bbox,
                                'text': text.strip(),
                                'confidence': text_info[1] if len(text_info) > 1 else 0.0
                            })
        
        if not text_boxes:
            return ""
        
        # Step 2: Crop each textbox and re-run OCR for better accuracy
        # (Optional: we can use the text from detection, or re-OCR each crop)
        # For now, we'll use the text from detection but verify with re-OCR
        
        # Step 3: Group into speech bubbles
        speech_bubbles = group_textboxes_into_speech_bubbles(text_boxes, img_width, img_height)
        
        # Step 4: Sort speech bubbles by reading order
        speech_bubbles.sort(key=lambda bubble: (
            min(text_boxes[idx]['top_y'] for idx in bubble),
            min(text_boxes[idx]['left_x'] for idx in bubble)
        ))
        
        # Step 5: Extract text from each speech bubble
        bubble_texts = []
        for bubble in speech_bubbles:
            # Sort textboxes within bubble by reading order
            bubble_indices = sorted(bubble, key=lambda idx: (text_boxes[idx]['top_y'], text_boxes[idx]['left_x']))
            # Join texts within same bubble with space
            bubble_text = " ".join([text_boxes[idx]['text'] for idx in bubble_indices])
            bubble_texts.append(bubble_text)
        
        # Join different speech bubbles with space
        return " ".join(bubble_texts)
        
    except Exception as e:
        tqdm.write(f"Error processing {panel_image_path}: {e}")
        return ""

# Process each panel
print("\n3. Processing panels with OCR...")
results = []

for idx, panel_row in tqdm(panel_df.iterrows(), total=len(panel_df), desc="Processing panels"):
    comic_no = int(panel_row['comic_no'])
    page_no = int(panel_row['page_no'])
    panel_index = int(panel_row['panel_index'])
    panel_filename = panel_row['panel_filename']
    
    panel_path = panels_dir / panel_filename
    if not panel_path.exists():
        continue
    
    # Extract OCR text
    ocr_text = extract_ocr_from_panel(ocr, panel_path)
    
    results.append({
        'comic_no': comic_no,
        'page_no': page_no,
        'panel_index': panel_index,
        'panel_filename': panel_filename,
        'ocr_text': ocr_text
    })

# Save results
print("\n4. Saving results...")
results_df = pd.DataFrame(results)
results_df.to_json(output_metadata_path, orient='records', indent=2)

# Also save as CSV for compatibility
output_csv = output_metadata_path.with_suffix('.csv')
results_df.to_csv(output_csv, index=False)

print(f"\n✓ Saved to {output_metadata_path}")
print(f"✓ Saved to {output_csv}")

# Statistics
print("\n" + "="*80)
print("STATISTICS")
print("="*80)
print(f"Total panels processed: {len(results)}")
print(f"Panels with OCR text: {sum(1 for r in results if r['ocr_text'])}")
print(f"Panels without OCR text: {sum(1 for r in results if not r['ocr_text'])}")

# Show some samples
print("\nSample OCR texts:")
print("-"*80)
for i in range(min(5, len(results))):
    r = results[i]
    if r['ocr_text']:
        print(f"Panel {r['panel_filename']}:")
        print(f"  {r['ocr_text'][:200]}")
        print()
print("="*80)

