#!/usr/bin/env python3
"""
Sample 30,000 panels from ~/data/raw_panel_images while preserving panel order
Panels from the same page are kept together to maintain triplet structure
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import random

# Configuration
SOURCE_DIR = Path(os.path.expanduser("~/data/raw_panel_images"))
TARGET_DIR = Path(os.path.expanduser("~/data/raw_panel_images_small"))
TARGET_COUNT = 30000

def get_page_info(panel_path):
    """Extract (comic_no, page_no) from panel path"""
    parts = panel_path.parts
    comic_no = None
    for part in parts:
        if part.isdigit() and part != 'raw_panel_images':
            comic_no = part
            break
    
    stem = panel_path.stem
    filename_parts = stem.split('_')
    if len(filename_parts) >= 2:
        page_no = filename_parts[0]
        return comic_no, page_no
    return None, None

def group_panels_by_page(source_dir):
    """Group all panels by (comic_no, page_no) to preserve order"""
    print("Scanning and grouping panels by page...")
    pages = defaultdict(list)
    
    for img_file in tqdm(source_dir.rglob("*.jpg"), desc="Scanning"):
        comic_no, page_no = get_page_info(img_file)
        if comic_no is not None and page_no is not None:
            pages[(comic_no, page_no)].append(img_file)
    
    # Sort panels within each page by panel index
    for page_key in pages:
        pages[page_key].sort(key=lambda p: int(p.stem.split('_')[1]) if len(p.stem.split('_')) >= 2 else 0)
    
    return pages

def sample_pages(pages, target_count):
    """Sample pages until we reach target panel count"""
    print(f"\nSampling pages to get ~{target_count} panels...")
    
    # Calculate panels per page
    page_list = [(key, len(panels)) for key, panels in pages.items()]
    page_list.sort(key=lambda x: x[1], reverse=True)  # Sort by panel count
    
    selected_pages = []
    total_panels = 0
    
    # Random seed for reproducibility
    random.seed(42)
    page_list_shuffled = page_list.copy()
    random.shuffle(page_list_shuffled)
    
    for (comic_no, page_no), panel_count in tqdm(page_list_shuffled, desc="Selecting pages"):
        if total_panels + panel_count <= target_count:
            selected_pages.append((comic_no, page_no))
            total_panels += panel_count
        elif total_panels < target_count:
            # If adding this page would exceed, check if it's close enough
            if target_count - total_panels < panel_count * 0.5:  # If less than half the page
                selected_pages.append((comic_no, page_no))
                total_panels += panel_count
                break
    
    print(f"Selected {len(selected_pages)} pages with {total_panels} panels")
    return selected_pages, total_panels

def copy_panels(pages, selected_pages, source_dir, target_dir):
    """Copy selected panels to target directory maintaining structure"""
    print(f"\nCopying panels to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    for comic_no, page_no in tqdm(selected_pages, desc="Copying"):
        page_key = (comic_no, page_no)
        if page_key not in pages:
            continue
        
        # Create comic_no directory in target
        comic_dir = target_dir / comic_no
        comic_dir.mkdir(exist_ok=True)
        
        # Copy all panels from this page
        for panel_path in pages[page_key]:
            target_path = comic_dir / panel_path.name
            shutil.copy2(panel_path, target_path)
            copied_count += 1
    
    print(f"Copied {copied_count} panels")
    return copied_count

def main():
    print("="*80)
    print("Sampling 30,000 panels while preserving panel order")
    print("="*80)
    
    # Step 1: Group panels by page
    pages = group_panels_by_page(SOURCE_DIR)
    print(f"\nFound {len(pages)} unique pages")
    
    # Step 2: Sample pages
    selected_pages, expected_count = sample_pages(pages, TARGET_COUNT)
    
    # Step 3: Copy panels
    copied_count = copy_panels(pages, selected_pages, SOURCE_DIR, TARGET_DIR)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Target directory: {TARGET_DIR}")
    print(f"Total pages in source: {len(pages)}")
    print(f"Selected pages: {len(selected_pages)}")
    print(f"Expected panels: {expected_count}")
    print(f"Copied panels: {copied_count}")
    print(f"Target count: {TARGET_COUNT}")
    print("="*80)

if __name__ == "__main__":
    main()

