#!/usr/bin/env python3
"""
Label panel reading order for the large dataset (1.2M panels)
This script assumes panels are already cropped and we need to add reading order labels
"""
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

def get_panel_info_from_path(panel_path):
    """
    Extract panel information from path structure
    Expected structure: ~/data/raw_panel_images/{comic_no}/{page_no}_{panel_idx}.jpg
    """
    panel_path = Path(panel_path)
    parts = panel_path.parts
    
    # Find comic_no (directory name)
    comic_no = None
    for part in parts:
        if part.isdigit() and part != 'raw_panel_images':
            comic_no = int(part)
            break
    
    # Extract page_no and panel_idx from filename: {page_no}_{panel_idx}.jpg
    stem = panel_path.stem
    filename_parts = stem.split('_')
    if len(filename_parts) >= 2:
        try:
            page_no = int(filename_parts[0])
            panel_idx = int(filename_parts[1])
            return comic_no, page_no, panel_idx
        except ValueError:
            pass
    
    return comic_no, None, None

def process_large_dataset(data_dir, output_dir, metadata_file=None):
    """
    Process large dataset: add reading order labels to panels
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all panel images (optimized for large dataset)
    print("Scanning for panel images...")
    panel_extensions = ['.jpg', '.jpeg', '.png']
    panel_files = []
    for ext in panel_extensions:
        # Use glob for better performance
        panel_files.extend(list(data_dir.rglob(f'*{ext}')))
    
    print(f"Found {len(panel_files)} panel images")
    
    # Group panels by (comic_no, page_no)
    panels_by_page = {}
    
    for panel_path in tqdm(panel_files, desc="Grouping panels"):
        comic_no, page_no, panel_idx = get_panel_info_from_path(panel_path)
        
        if comic_no is not None and page_no is not None:
            key = (comic_no, page_no)
            if key not in panels_by_page:
                panels_by_page[key] = []
            
            panels_by_page[key].append({
                'path': str(panel_path),
                'panel_idx': panel_idx if panel_idx is not None else -1,
                'filename': panel_path.name
            })
    
    # Sort panels within each page and assign reading order
    metadata = []
    
    for (comic_no, page_no), panels in tqdm(panels_by_page.items(), desc="Labeling order"):
        # Sort panels by panel_idx if available, otherwise by filename
        if all(p['panel_idx'] >= 0 for p in panels):
            sorted_panels = sorted(panels, key=lambda p: p['panel_idx'])
        else:
            # Sort by filename (lexicographic order)
            sorted_panels = sorted(panels, key=lambda p: p['filename'])
        
        # Assign reading order (0-indexed)
        for reading_order, panel in enumerate(sorted_panels):
            metadata.append({
                'comic_no': comic_no,
                'page_no': page_no,
                'panel_index': reading_order,  # Reading order in page
                'panel_path': panel['path'],
                'panel_filename': panel['filename'],
                'total_panels_in_page': len(sorted_panels)
            })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_path = output_dir / 'panel_order_metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"\nSaved metadata to {metadata_path}")
    print(f"Total panels labeled: {len(metadata)}")
    print(f"Total pages: {metadata_df[['comic_no', 'page_no']].drop_duplicates().shape[0]}")
    
    return metadata_df

if __name__ == "__main__":
    data_dir = os.path.expanduser("~/data/raw_panel_images")
    output_dir = os.path.expanduser("~/data/processed_large_dataset")
    
    print(f"Processing large dataset from: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    metadata_df = process_large_dataset(data_dir, output_dir)
    print("\nDone!")

