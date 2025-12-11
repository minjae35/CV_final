#!/usr/bin/env python3
"""
Crop panels from page images and label them with reading order (left->right, top->bottom)
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

def parse_annotation_file(ann_path):
    """
    Parse annotation file to get panel bounding boxes
    Format: class_id x1 y1 x2 y2 (one per line)
    class_id=1 means panel
    """
    panels = []
    with open(ann_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                if class_id == 1:  # Panel
                    x1, y1, x2, y2 = map(int, parts[1:5])
                    panels.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'center_x': (x1 + x2) / 2,
                        'center_y': (y1 + y2) / 2
                    })
    return panels

def sort_panels_by_reading_order(panels):
    """
    Sort panels by reading order: top->bottom, then left->right
    Uses a simple heuristic: sort by y-coordinate first (with tolerance), then by x-coordinate
    """
    # Group panels by approximate row (y-coordinate with tolerance)
    tolerance = 100  # pixels tolerance for same row (increased for better grouping)
    
    # Sort by y-coordinate first (top to bottom), then by x-coordinate (left to right)
    sorted_panels = sorted(panels, key=lambda p: (p['y1'] // tolerance, p['x1']))
    
    return sorted_panels

def crop_panel(image, bbox):
    """Crop panel from image using bounding box"""
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return image[y1:y2, x1:x2]

def process_444_pages(data_dir, output_dir):
    """
    Process 444 pages: crop panels and label with reading order
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    annotations_dir = data_dir / 'panels_annotations' / 'Annotations'
    images_dir = data_dir / 'panels_annotations' / 'Images'
    output_panels_dir = output_dir / 'cropped_panels'
    output_panels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all annotation files
    ann_files = sorted(annotations_dir.glob('*.txt'))
    
    metadata = []
    
    print(f"Processing {len(ann_files)} pages...")
    
    for ann_file in tqdm(ann_files):
        # Parse filename: {comic_no}_{page_no}.txt
        stem = ann_file.stem
        try:
            comic_no, page_no = stem.split('_')
        except ValueError:
            print(f"Warning: Invalid filename format {stem}, skipping")
            continue
        
        # Load corresponding image
        img_path = images_dir / f"{stem}.jpg"
        if not img_path.exists():
            print(f"Warning: Image not found for {stem}")
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            continue
        
        # Parse annotations
        panels = parse_annotation_file(ann_file)
        if len(panels) == 0:
            print(f"Warning: No panels found in {stem}")
            continue
        
        # Sort panels by reading order
        sorted_panels = sort_panels_by_reading_order(panels)
        
        # Crop and save each panel
        for panel_idx, panel in enumerate(sorted_panels):
            cropped = crop_panel(image, panel)
            if cropped is None:
                continue
            
            # Save cropped panel
            panel_filename = f"{comic_no}_{page_no}_panel_{panel_idx:02d}.jpg"
            panel_path = output_panels_dir / panel_filename
            cv2.imwrite(str(panel_path), cropped)
            
            # Store metadata
            metadata.append({
                'comic_no': int(comic_no),
                'page_no': int(page_no),
                'panel_index': panel_idx,  # Reading order (0-indexed)
                'panel_filename': panel_filename,
                'original_bbox': [panel['x1'], panel['y1'], panel['x2'], panel['y2']],
                'total_panels_in_page': len(sorted_panels)
            })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_path = output_dir / 'panel_metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\nSaved metadata to {metadata_path}")
    print(f"Total panels cropped: {len(metadata)}")
    print(f"Total pages processed: {metadata_df[['comic_no', 'page_no']].drop_duplicates().shape[0]}")
    
    return metadata_df

if __name__ == "__main__":
    data_dir = "/home/mjb4835/CV_final_project/data"
    output_dir = "/home/mjb4835/CV_final_project/data/processed_444_pages"
    
    metadata_df = process_444_pages(data_dir, output_dir)
    print("\nDone!")

