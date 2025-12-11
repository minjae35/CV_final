"""
PanelDataset and TripletDataset for Visual Narrative Understanding
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_default_transform(image_size: int = 224, is_train: bool = True):
    """
    Get default image transformation
    
    Args:
        image_size: Target image size
        is_train: If True, apply augmentation; if False, only normalization
    
    Returns:
        transforms.Compose
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class PanelDataset(Dataset):
    """
    Dataset for loading individual panels (image + OCR text)
    """
    def __init__(
        self,
        metadata_csv_path: Union[str, Path],
        panels_dir: Union[str, Path],
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            metadata_csv_path: Path to panel_metadata_with_ocr.csv
            panels_dir: Directory containing panel images
            transform: Optional image transformation
        """
        self.metadata_csv_path = Path(metadata_csv_path)
        self.panels_dir = Path(panels_dir)
        self.transform = transform
        
        # Load metadata
        self.df = pd.read_csv(self.metadata_csv_path)
        print(f"Loaded {len(self.df)} panels from {self.metadata_csv_path}")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dict with keys: 'image' (Tensor), 'ocr_text' (str), 'metadata' (Dict)
        """
        row = self.df.iloc[idx]
        
        # Load image
        panel_filename = row['panel_filename']
        image_path = self._find_image_path(row['comic_no'], row['page_no'], row['panel_index'], panel_filename)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        
        # Get OCR text
        ocr_text = str(row.get('ocr_text', '')) if pd.notna(row.get('ocr_text')) else ''
        if not ocr_text.strip():
            ocr_text = "[EMPTY]"
        
        return {
            'image': image,
            'ocr_text': ocr_text,
            'metadata': {
                'comic_no': int(row['comic_no']),
                'page_no': int(row['page_no']),
                'panel_index': int(row['panel_index']),
                'panel_filename': panel_filename
            }
        }
    
    def _find_image_path(self, comic_no: int, page_no: int, panel_index: int, panel_filename: str) -> Path:
        """
        Find image path based on dataset structure
        Handles both flat structure (444 dataset) and nested structure (30k dataset)
        """
        # Try direct filename first (for 444 dataset: cropped_panels/0_19_panel_00.jpg)
        direct_path = self.panels_dir / panel_filename
        if direct_path.exists():
            return direct_path
        
        # Try nested structure (for 30k dataset: raw_panel_images_small/{comic_no}/{page_no}_{panel_index}.jpg)
        nested_path = self.panels_dir / str(comic_no) / panel_filename
        if nested_path.exists():
            return nested_path
        
        # Fallback: try various patterns
        # Pattern: {comic_no}/{page_no}_{panel_index}.jpg
        fallback_path = self.panels_dir / str(comic_no) / f"{page_no}_{panel_index}.jpg"
        if fallback_path.exists():
            return fallback_path
        
        # If still not found, return the first attempt (will raise error)
        return direct_path


class TripletDataset(Dataset):
    """
    Dataset for loading triplets (A, B, C panels + Hard Negative candidates)
    """
    def __init__(
        self,
        triplets_json_path: Union[str, Path],
        panels_dir: Union[str, Path],
        metadata_csv_path: Optional[Union[str, Path]] = None,
        transform: Optional[transforms.Compose] = None,
        return_pil: bool = False
    ):
        """
        Args:
            triplets_json_path: Path to triplets JSON file
            panels_dir: Directory containing panel images
            metadata_csv_path: Optional path to metadata CSV (for faster OCR lookup)
            transform: Optional image transformation
            return_pil: If True, return PIL images instead of tensors
        """
        self.triplets_json_path = Path(triplets_json_path)
        self.panels_dir = Path(panels_dir)
        self.transform = transform
        self.return_pil = return_pil
        
        # Load triplets
        print(f"Loading triplets from {self.triplets_json_path}...")
        with open(self.triplets_json_path, 'r', encoding='utf-8') as f:
            self.triplets = json.load(f)
        print(f"Loaded {len(self.triplets)} triplets")
        
        # Optionally load metadata for faster OCR lookup
        self.metadata_df = None
        if metadata_csv_path:
            metadata_path = Path(metadata_csv_path)
            if metadata_path.exists():
                self.metadata_df = pd.read_csv(metadata_path)
                print(f"Loaded metadata from {metadata_path}")
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dict with keys:
                - 'A': Dict with 'image', 'ocr_text', 'metadata'
                - 'B': Dict with 'image', 'ocr_text', 'metadata'
                - 'C': Dict with 'image', 'ocr_text', 'metadata'
                - 'neg_candidates': List of Dict with 'image', 'ocr_text', 'metadata'
        """
        triplet = self.triplets[idx]
        
        # Load A, B, C panels
        panel_a = self._load_panel(triplet['A'])
        panel_b = self._load_panel(triplet['B'])
        panel_c = self._load_panel(triplet['C'])
        
        # Load negative candidates
        neg_candidates = []
        for neg_info in triplet.get('neg_candidates', []):
            neg_panel = self._load_panel(neg_info)
            neg_candidates.append(neg_panel)
        
        return {
            'A': panel_a,
            'B': panel_b,
            'C': panel_c,
            'neg_candidates': neg_candidates
        }
    
    def _load_panel(self, panel_info: Dict) -> Dict:
        """
        Load a single panel (image + OCR text)
        
        Args:
            panel_info: Dict with 'comic_no', 'page_no', 'panel_index', 'panel_filename', 'ocr_text'
        
        Returns:
            Dict with 'image' (Tensor or PIL), 'pil_image' (PIL if return_pil=True), 'ocr_text' (str), 'metadata' (Dict)
        """
        comic_no = panel_info['comic_no']
        page_no = panel_info['page_no']
        panel_index = panel_info['panel_index']
        panel_filename = panel_info.get('panel_filename', '')
        ocr_text = panel_info.get('ocr_text', '')
        
        # Find image path
        image_path = self._find_image_path(comic_no, page_no, panel_index, panel_filename)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        pil_image = Image.open(image_path).convert('RGB')
        
        # Apply transform or keep PIL
        if self.return_pil:
            # Return PIL image for CLIP preprocessing
            image = pil_image
        elif self.transform:
            image = self.transform(pil_image)
        else:
            to_tensor = transforms.ToTensor()
            image = to_tensor(pil_image)
        
        # Process OCR text
        if not ocr_text or (isinstance(ocr_text, str) and len(ocr_text.strip()) == 0):
            ocr_text = "[EMPTY]"
        else:
            ocr_text = str(ocr_text)
        
        result = {
            'image': image,
            'ocr_text': ocr_text,
            'metadata': {
                'comic_no': comic_no,
                'page_no': page_no,
                'panel_index': panel_index,
                'panel_filename': panel_filename
            }
        }
        
        # Add PIL image if requested
        if self.return_pil:
            result['pil_image'] = pil_image
        
        return result
    
    def _find_image_path(self, comic_no: int, page_no: int, panel_index: int, panel_filename: str) -> Path:
        """
        Find image path based on dataset structure
        Handles both flat structure (444 dataset) and nested structure (30k dataset)
        """
        # Try direct filename first (for 444 dataset: cropped_panels/0_19_panel_00.jpg)
        if panel_filename:
            direct_path = self.panels_dir / panel_filename
            if direct_path.exists():
                return direct_path
        
        # Try nested structure (for 30k dataset: raw_panel_images_small/{comic_no}/{page_no}_{panel_index}.jpg)
        if panel_filename:
            nested_path = self.panels_dir / str(comic_no) / panel_filename
            if nested_path.exists():
                return nested_path
        
        # Fallback: try pattern {comic_no}/{page_no}_{panel_index}.jpg
        fallback_path = self.panels_dir / str(comic_no) / f"{page_no}_{panel_index}.jpg"
        if fallback_path.exists():
            return fallback_path
        
        # If still not found, try direct with constructed filename
        if not panel_filename:
            # Try to construct filename
            direct_path = self.panels_dir / f"{page_no}_{panel_index}.jpg"
            if direct_path.exists():
                return direct_path
        
        # Last resort: return expected path (will raise error if not found)
        return self.panels_dir / panel_filename if panel_filename else self.panels_dir / f"{page_no}_{panel_index}.jpg"


def collate_triplets(batch: List[Dict]) -> Dict:
    """
    Collate function for TripletDataset
    
    Args:
        batch: List of triplet dictionaries
    
    Returns:
        Batched dictionary with stacked tensors
    """
    # Extract A, B, C images and texts
    images_a = torch.stack([item['A']['image'] for item in batch])
    images_b = torch.stack([item['B']['image'] for item in batch])
    images_c = torch.stack([item['C']['image'] for item in batch])
    
    ocr_texts_a = [item['A']['ocr_text'] for item in batch]
    ocr_texts_b = [item['B']['ocr_text'] for item in batch]
    ocr_texts_c = [item['C']['ocr_text'] for item in batch]
    
    # Metadata
    metadata_a = [item['A']['metadata'] for item in batch]
    metadata_b = [item['B']['metadata'] for item in batch]
    metadata_c = [item['C']['metadata'] for item in batch]
    
    # Negative candidates (multiple negatives per sample)
    neg_candidates_list = [item['neg_candidates'] for item in batch]
    
    # Handle multiple negatives: flatten all negatives into a single batch
    # Format: [batch_size * num_negatives, C, H, W]
    neg_images = []
    neg_texts = []
    neg_metadata = []
    neg_batch_indices = []  # Track which batch item each negative belongs to
    
    for batch_idx, neg_cands in enumerate(neg_candidates_list):
        if len(neg_cands) > 0:
            for neg in neg_cands:
                neg_images.append(neg['image'])
                neg_texts.append(neg['ocr_text'])
                neg_metadata.append(neg['metadata'])
                neg_batch_indices.append(batch_idx)
        else:
            # If no negative, use a placeholder (B panel)
            neg_images.append(images_b[batch_idx])
            neg_texts.append("[EMPTY]")
            neg_metadata.append({})
            neg_batch_indices.append(batch_idx)
    
    if neg_images:
        neg_images = torch.stack(neg_images)
    else:
        neg_images = torch.empty(0)
    
    return {
        'A': {
            'image': images_a,
            'ocr_text': ocr_texts_a,
            'metadata': metadata_a
        },
        'B': {
            'image': images_b,
            'ocr_text': ocr_texts_b,
            'metadata': metadata_b
        },
        'C': {
            'image': images_c,
            'ocr_text': ocr_texts_c,
            'metadata': metadata_c
        },
        'neg_candidates': {
            'image': neg_images,
            'ocr_text': neg_texts,
            'metadata': neg_metadata,
            'batch_indices': neg_batch_indices  # Track which batch item each negative belongs to
        }
    }

