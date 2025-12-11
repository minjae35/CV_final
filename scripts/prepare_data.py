#!/usr/bin/env python3
"""
Data preparation script: OCR text matching and triplet generation
"""
import os
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def match_ocr_to_panels_from_files(panel_metadata_path, ocr_dir):
    """
    Match OCR text to panels from OCR result files (.txt)
    
    Args:
        panel_metadata_path: Path to panel_metadata.csv
        ocr_dir: Directory containing OCR result .txt files
    
    Returns:
        DataFrame with panel metadata + OCR text
    """
    print(f"Loading panel metadata from {panel_metadata_path}...")
    panel_df = pd.read_csv(panel_metadata_path)
    print(f"Loaded {len(panel_df)} panels")
    
    ocr_dir = Path(ocr_dir)
    ocr_texts = []
    
    for idx, panel_row in tqdm(panel_df.iterrows(), total=len(panel_df), desc="Loading OCR from files"):
        comic_no = panel_row.get('comic_no', None)
        page_no = panel_row.get('page_no', None)
        panel_index = panel_row.get('panel_index', None)
        panel_filename = panel_row['panel_filename']
        
        # Try multiple OCR file naming patterns
        ocr_path = None
        
        # Pattern 1: {comic_no}_{page_no}_panel_{panel_index:02d}.txt (for processed_444_pages)
        if comic_no is not None and page_no is not None and panel_index is not None:
            ocr_filename_1 = f"{comic_no}_{page_no}_panel_{panel_index:02d}.txt"
            ocr_path_1 = ocr_dir / ocr_filename_1
            if ocr_path_1.exists():
                ocr_path = ocr_path_1
        
        # Pattern 2: {comic_no}_{panel_filename_base}.txt (for raw_panel_images_small)
        if ocr_path is None and comic_no is not None:
            base = panel_filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
            # Check if panel_filename already contains comic_no
            if not str(panel_filename).startswith(str(comic_no)):
                ocr_filename_2 = f"{comic_no}_{base}.txt"
                ocr_path_2 = ocr_dir / ocr_filename_2
                if ocr_path_2.exists():
                    ocr_path = ocr_path_2
        
        # Pattern 3: Just replace extension (fallback)
        if ocr_path is None:
            ocr_filename_3 = panel_filename.replace('.jpg', '.txt').replace('.png', '.txt')
            ocr_path_3 = ocr_dir / ocr_filename_3
            if ocr_path_3.exists():
                ocr_path = ocr_path_3
        
        # Read OCR text if file exists
        if ocr_path is not None and ocr_path.exists():
            ocr_text = ocr_path.read_text(encoding='utf-8').strip()
            # Replace newlines and multiple whitespaces with single space
            ocr_text = ' '.join(ocr_text.split())
            ocr_texts.append(ocr_text)
        else:
            # OCR file not found, use empty string
            ocr_texts.append("")
    
    panel_df['ocr_text'] = ocr_texts
    panels_with_text = panel_df['ocr_text'].str.len().gt(0).sum()
    
    print(f"Loaded OCR for {len(panel_df)} panels")
    print(f"  - Panels with text: {panels_with_text}")
    print(f"  - Panels without text: {len(panel_df) - panels_with_text}")
    
    return panel_df


def generate_triplets(panel_df):
    """
    Generate (A, B, C) triplets from panel metadata
    
    Args:
        panel_df: DataFrame with panel metadata (must have comic_no, page_no, panel_index)
    
    Returns:
        List of triplets: [(A_info, B_info, C_info, neg_candidates), ...]
    """
    print("Generating triplets...")
    
    # Group panels by (comic_no, page_no)
    panels_by_page = defaultdict(list)
    for _, row in panel_df.iterrows():
        key = (int(row['comic_no']), int(row['page_no']))
        panels_by_page[key].append(row)
    
    triplets = []
    
    # Collect all panels for cross-page negative sampling
    all_panels_list = []
    for (comic_no, page_no), panels in panels_by_page.items():
        all_panels_list.extend(panels)
    
    for (comic_no, page_no), panels in tqdm(panels_by_page.items(), desc="Processing pages"):
        # Sort panels by panel_index
        panels_sorted = sorted(panels, key=lambda x: int(x['panel_index']))
        
        # Generate triplets: (P_i, P_{i+1}, P_{i+2})
        for i in range(len(panels_sorted) - 2):
            panel_a = panels_sorted[i]
            panel_b = panels_sorted[i + 1]
            panel_c = panels_sorted[i + 2]
            
            # Hard negative: select panels close to A-C range
            # Choose panels that are adjacent or near to A-C (more challenging)
            panel_a_idx = int(panel_a['panel_index'])
            panel_b_idx = int(panel_b['panel_index'])
            panel_c_idx = int(panel_c['panel_index'])
            
            # Strategy: Select 3-5 negatives with varying difficulty
            # 1. Adjacent panels (hardest): A-1, C+1, C+2
            # 2. Nearby panels: within distance 3-5
            # 3. Far panels: distance > 5 (if needed)
            
            candidates = []
            
            # First, try to find candidates from the same page
            for p in panels_sorted:
                p_idx = int(p['panel_index'])
                if p_idx != panel_a_idx and p_idx != panel_b_idx and p_idx != panel_c_idx:
                    if p_idx < panel_a_idx:
                        # Before A: distance from A
                        distance = panel_a_idx - p_idx
                        candidates.append((p, distance, 'before', 'same_page'))
                    elif p_idx > panel_c_idx:
                        # After C: distance from C
                        distance = p_idx - panel_c_idx
                        candidates.append((p, distance, 'after', 'same_page'))
            
            # If no candidates from same page, search in other pages (same comic)
            if not candidates:
                for p in all_panels_list:
                    p_comic = int(p['comic_no'])
                    p_page = int(p['page_no'])
                    p_idx = int(p['panel_index'])
                    
                    # Skip if same panel
                    if p_comic == comic_no and p_page == page_no and p_idx in [panel_a_idx, panel_b_idx, panel_c_idx]:
                        continue
                    
                    # Only use panels from the SAME comic (different page)
                    # This makes it harder than using completely different comics
                    if p_comic == comic_no and p_page != page_no:
                        distance = abs(p_page - page_no)
                        candidates.append((p, distance, 'cross_page', 'same_comic'))
            
            # Select multiple negatives with varying difficulty
            neg_candidates = []
            if candidates:
                # Sort by distance (ascending) to get closest panels first
                # Prioritize same-page candidates
                candidates.sort(key=lambda x: (x[3] != 'same_page', x[1]))
                
                # Select up to 3 negatives with different distances
                selected_distances = set()
                for p, dist, pos, src in candidates:
                    if len(neg_candidates) >= 3:
                        break
                    # Avoid selecting panels with the same distance
                    if dist not in selected_distances:
                        neg_candidates.append({
                            'comic_no': int(p['comic_no']),
                            'page_no': int(p['page_no']),
                            'panel_index': int(p['panel_index']),
                            'panel_filename': p.get('panel_filename', ''),
                            'ocr_text': p.get('ocr_text', '')
                        })
                        selected_distances.add(dist)
            
            triplet = {
                'A': {
                    'comic_no': int(panel_a['comic_no']),
                    'page_no': int(panel_a['page_no']),
                    'panel_index': int(panel_a['panel_index']),
                    'panel_filename': panel_a.get('panel_filename', ''),
                    'ocr_text': panel_a.get('ocr_text', '')
                },
                'B': {
                    'comic_no': int(panel_b['comic_no']),
                    'page_no': int(panel_b['page_no']),
                    'panel_index': int(panel_b['panel_index']),
                    'panel_filename': panel_b.get('panel_filename', ''),
                    'ocr_text': panel_b.get('ocr_text', '')
                },
                'C': {
                    'comic_no': int(panel_c['comic_no']),
                    'page_no': int(panel_c['page_no']),
                    'panel_index': int(panel_c['panel_index']),
                    'panel_filename': panel_c.get('panel_filename', ''),
                    'ocr_text': panel_c.get('ocr_text', '')
                },
                'neg_candidates': neg_candidates
            }
            triplets.append(triplet)
    
    print(f"Generated {len(triplets)} triplets")
    return triplets


def save_triplets(triplets, output_path):
    """Save triplets to JSON file"""
    print(f"Saving triplets to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(triplets, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(triplets)} triplets")


if __name__ == "__main__":
    # Configuration
    base_dir = Path("/home/mjb4835/CV_final_project")
    
    # 444 pages subset
    panel_metadata_444 = base_dir / "data" / "processed_444_pages" / "panel_metadata.csv"
    ocr_dir_444 = base_dir / "data" / "processed_444_pages" / "ocr_result"
    output_metadata_444 = base_dir / "data" / "processed_444_pages" / "panel_metadata_with_ocr.csv"
    output_triplets_444 = base_dir / "data" / "processed_444_pages" / "triplets_444.json"
    
    # Process 444 pages subset
    print("\n" + "="*50)
    print("Processing 444 pages subset")
    print("="*50)
    
    if panel_metadata_444.exists() and ocr_dir_444.exists():
        print(f"Using OCR result files from {ocr_dir_444}")
        panel_df_444 = match_ocr_to_panels_from_files(panel_metadata_444, ocr_dir_444)
        
        panel_df_444.to_csv(output_metadata_444, index=False)
        print(f"Saved panel metadata with OCR to {output_metadata_444}")
        
        triplets_444 = generate_triplets(panel_df_444)
        save_triplets(triplets_444, output_triplets_444)
        
        # Statistics
        print("\n" + "="*50)
        print("Statistics (444 pages subset)")
        print("="*50)
        print(f"Total panels: {len(panel_df_444)}")
        print(f"Total triplets: {len(triplets_444)}")
        print(f"Average triplets per page: {len(triplets_444) / panel_df_444[['comic_no', 'page_no']].drop_duplicates().shape[0]:.2f}")
        print(f"Panels with OCR text: {panel_df_444['ocr_text'].str.len().gt(0).sum()}")
    else:
        print(f"Warning: Required files not found. Skipping 444 pages subset processing.")
    
    # Process raw_panel_images_small dataset (30K panels)
    print("\n" + "="*50)
    print("Processing raw_panel_images_small dataset (~30K panels)")
    print("="*50)
    
    panel_metadata_small = Path(os.path.expanduser("~/data/raw_panel_images_small_metadata.csv"))
    ocr_dir_small = Path(os.path.expanduser("~/data/small_ocr_result"))
    output_metadata_small = Path(os.path.expanduser("~/data/raw_panel_images_small_metadata_with_ocr.csv"))
    output_triplets_small = Path(os.path.expanduser("~/data/triplets_small.json"))
    
    if panel_metadata_small.exists() and ocr_dir_small.exists():
        print(f"Using OCR result files from {ocr_dir_small}")
        panel_df_small = match_ocr_to_panels_from_files(panel_metadata_small, ocr_dir_small)
        
        panel_df_small.to_csv(output_metadata_small, index=False)
        print(f"Saved panel metadata with OCR to {output_metadata_small}")
        
        triplets_small = generate_triplets(panel_df_small)
        save_triplets(triplets_small, output_triplets_small)
        
        # Statistics
        print("\n" + "="*50)
        print("Statistics (raw_panel_images_small dataset)")
        print("="*50)
        print(f"Total panels: {len(panel_df_small)}")
        print(f"Total triplets: {len(triplets_small)}")
        print(f"Average triplets per page: {len(triplets_small) / panel_df_small[['comic_no', 'page_no']].drop_duplicates().shape[0]:.2f}")
        print(f"Panels with OCR text: {panel_df_small['ocr_text'].str.len().gt(0).sum()}")
    else:
        print(f"Warning: Required files not found. Skipping raw_panel_images_small processing.")
    
    # Process large dataset (1.2M panels)
    print("\n" + "="*50)
    print("Processing large dataset (1.2M panels)")
    print("="*50)
    
    panel_metadata_large = Path(os.path.expanduser("~/data/raw_panel_images_metadata.csv"))
    ocr_dir_large = Path(os.path.expanduser("~/data/large_ocr_result"))
    output_metadata_large = Path(os.path.expanduser("~/data/raw_panel_images_metadata_with_ocr.csv"))
    output_triplets_large = Path(os.path.expanduser("~/data/triplets_large.json"))
    
    if panel_metadata_large.exists() and ocr_dir_large.exists():
        print(f"Using OCR result files from {ocr_dir_large}")
        panel_df_large = match_ocr_to_panels_from_files(panel_metadata_large, ocr_dir_large)
        
        panel_df_large.to_csv(output_metadata_large, index=False)
        print(f"Saved panel metadata with OCR to {output_metadata_large}")
        
        triplets_large = generate_triplets(panel_df_large)
        save_triplets(triplets_large, output_triplets_large)
        
        # Statistics
        print("\n" + "="*50)
        print("Statistics (Large dataset)")
        print("="*50)
        print(f"Total panels: {len(panel_df_large)}")
        print(f"Total triplets: {len(triplets_large)}")
        print(f"Average triplets per page: {len(triplets_large) / panel_df_large[['comic_no', 'page_no']].drop_duplicates().shape[0]:.2f}")
        print(f"Panels with OCR text: {panel_df_large['ocr_text'].str.len().gt(0).sum()}")
    else:
        print(f"Warning: Required files not found. Skipping large dataset processing.")

