"""
Collect ALL Teacher Scores (Positive + Negatives) with BATCH PROCESSING
"""
import json
import argparse
import torch
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# CoT Prompt Template
SYSTEM_PROMPT = """You are an expert in analyzing comic panel sequences. 
Evaluate the coherence of three consecutive panels (A, B, C)."""

COT_PROMPT_TEMPLATE = """
Analyze the coherence of these three comic panels step by step:

1. **Visual continuity**: Are characters, backgrounds, and art styles consistent?
2. **Temporal flow**: Do actions progress logically in time?
3. **Narrative logic**: Is there clear cause-and-effect? Does the story make sense?
4. **Overall coherence**: How natural and coherent is the sequence?

Output in JSON format:
{
  "visual_continuity": "your analysis",
  "temporal_flow": "your analysis",
  "narrative_logic": "your analysis",
  "overall_coherence": "your analysis",
  "score": 0.95
}

Score from 0.0 (incoherent) to 0.95 (perfectly coherent).
"""


def load_qwen_model(device: str = "cuda"):
    """Load Qwen2.5-VL-3B model (FP16)"""
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        print("Loading Qwen2.5-VL-3B model (FP16)...")
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Fix padding side for decoder-only models (batch processing)
        processor.tokenizer.padding_side = 'left'
        
        print(f"✅ Model loaded (FP16, left-padding)")
        return model, processor
    
    except Exception as e:
        print(f"❌ Error loading Qwen model: {e}")
        return None, None


def find_panel_image(panels_dir: Path, panel_info: Dict) -> Optional[Path]:
    """Find panel image path"""
    panel_filename = panel_info.get('panel_filename', '')
    comic_no = panel_info['comic_no']
    
    direct_path = panels_dir / panel_filename
    if direct_path.exists():
        return direct_path
    
    nested_path = panels_dir / str(comic_no) / panel_filename
    if nested_path.exists():
        return nested_path
    
    return None


def query_teacher_model_batch(model, processor, panel_paths_batch: List[Dict], batch_size: int) -> List[Dict]:
    """Query teacher model with batch of triplets"""
    
    results = []
    
    try:
        from qwen_vl_utils import process_vision_info
        
        # Prepare all messages
        all_messages = []
        for paths in panel_paths_batch:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": str(paths['A'])},
                    {"type": "text", "text": "Panel A"},
                    {"type": "image", "image": str(paths['B'])},
                    {"type": "text", "text": "Panel B"},
                    {"type": "image", "image": str(paths['C'])},
                    {"type": "text", "text": "Panel C\n\n" + SYSTEM_PROMPT + "\n\n" + COT_PROMPT_TEMPLATE}
                ]
            }]
            all_messages.append(messages)
        
        # Process batch
        texts = []
        all_image_inputs = []
        all_video_inputs = []
        
        for messages in all_messages:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            
            image_inputs, video_inputs = process_vision_info(messages)
            if image_inputs:
                all_image_inputs.extend(image_inputs)
            if video_inputs:
                all_video_inputs.extend(video_inputs)
        
        # Batch process
        inputs = processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )
        
        # Decode each response
        input_token_len = inputs['input_ids'].shape[1]
        for i, output_id in enumerate(output_ids):
            response_text = processor.decode(
                output_id[input_token_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Parse JSON
            try:
                if '```json' in response_text:
                    json_str = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    json_str = response_text.split('```')[1].split('```')[0].strip()
                else:
                    json_str = response_text.strip()
                
                analysis = json.loads(json_str)
                results.append(analysis)
            except:
                results.append(None)
    
    except Exception as e:
        print(f"\n❌ Batch processing error: {e}")
        results = [None] * len(panel_paths_batch)
    
    return results


def collect_all_scores(
    triplets_json: Path,
    panels_dir: Path,
    output_json: Path,
    model,
    processor,
    num_samples: Optional[int] = None,
    offset: int = 0,
    batch_size: int = 2,
    save_every: int = 25,
    max_negatives: int = 2,
    collect_positive: bool = True,
    collect_negatives: bool = True
):
    """Collect teacher scores for both positive and negative samples"""
    
    # Load triplets
    print(f"Loading triplets from {triplets_json}...")
    with open(triplets_json, 'r') as f:
        data = json.load(f)
    
    # Handle both formats: dict with 'triplets' key or direct list
    if isinstance(data, dict) and 'triplets' in data:
        triplets = data['triplets']
    elif isinstance(data, list):
        triplets = data
    else:
        raise ValueError(f"Unexpected triplets format: {type(data)}")
    
    # Apply offset
    if offset > 0:
        triplets = triplets[offset:]
        print(f"Starting from offset {offset}")
    
    if num_samples:
        triplets = triplets[:num_samples]
    
    # Resume from existing results
    results = []
    failed_count = 0
    start_idx = 0
    
    if output_json.exists():
        print(f"📂 Found existing results, resuming...")
        with open(output_json, 'r') as f:
            existing_data = json.load(f)
            results = existing_data.get('results', [])
            failed_count = existing_data.get('num_failed', 0)
            # Count completed triplets
            completed_triplets = set()
            for r in results:
                key = (r['triplet_index'], r['is_positive'])
                completed_triplets.add(key)
            start_idx = len(completed_triplets)
        print(f"✅ Resuming from triplet {start_idx} ({len(results)} already collected)")
    
    # Prepare all samples (positive + negatives)
    all_samples = []
    for idx, triplet in enumerate(triplets):
        # Positive sample
        if collect_positive:
            all_samples.append({
                'triplet_index': offset + idx,
                'triplet': triplet,
                'is_positive': True,
                'panel_C': triplet['C']
            })
        
        # Negative samples
        if collect_negatives and 'neg_candidates' in triplet:
            neg_candidates = triplet['neg_candidates'][:max_negatives]
            for neg in neg_candidates:
                all_samples.append({
                    'triplet_index': offset + idx,
                    'triplet': triplet,
                    'is_positive': False,
                    'panel_C': neg
                })
    
    # Filter already completed
    if start_idx > 0:
        all_samples = all_samples[start_idx * (1 + max_negatives * collect_negatives):]
    
    print(f"Collecting scores for {len(all_samples)} samples (batch_size={batch_size})...")
    print(f"  - Positive: {collect_positive}, Negatives: {collect_negatives} (max {max_negatives})")
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    # Process in batches
    for i in tqdm(range(0, len(all_samples), batch_size), desc="Processing batches"):
        batch_samples = all_samples[i:i+batch_size]
        batch_paths = []
        batch_valid = []
        
        # Prepare batch
        for sample in batch_samples:
            panel_a_path = find_panel_image(panels_dir, sample['triplet']['A'])
            panel_b_path = find_panel_image(panels_dir, sample['triplet']['B'])
            panel_c_path = find_panel_image(panels_dir, sample['panel_C'])
            
            if all([panel_a_path, panel_b_path, panel_c_path]):
                batch_paths.append({
                    'A': panel_a_path,
                    'B': panel_b_path,
                    'C': panel_c_path
                })
                batch_valid.append(sample)
            else:
                failed_count += 1
        
        if not batch_paths:
            continue
        
        # Query batch
        batch_results = query_teacher_model_batch(model, processor, batch_paths, batch_size)
        
        # Store results
        for sample, analysis in zip(batch_valid, batch_results):
            if analysis:
                results.append({
                    'triplet_index': sample['triplet_index'],
                    'is_positive': sample['is_positive'],
                    'triplet': sample['triplet'],
                    'panel_C_used': sample['panel_C'],
                    'teacher_analysis': analysis,
                    'teacher_score': analysis.get('score', 0.5)
                })
            else:
                failed_count += 1
        
        # Periodic save
        batch_num = i // batch_size
        if (batch_num + 1) % save_every == 0:
            output_data = {
                'num_samples': len(results),
                'num_failed': failed_count,
                'batch_size': batch_size,
                'results': results
            }
            with open(output_json, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 Checkpoint: {len(results)} scores saved")
    
    # Final save
    output_data = {
        'num_samples': len(results),
        'num_failed': failed_count,
        'batch_size': batch_size,
        'results': results
    }
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Collected {len(results)} scores ({failed_count} failed)")
    print(f"Results saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description='Collect ALL Teacher Scores (Positive + Negatives)')
    parser.add_argument('--triplets', type=str,
                       default='data/processed_444_pages/triplets_444.json')
    parser.add_argument('--panels_dir', type=str,
                       default='data/processed_444_pages/cropped_panels')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_negatives', type=int, default=2)
    parser.add_argument('--positive_only', action='store_true')
    parser.add_argument('--negatives_only', action='store_true')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_qwen_model(device)
    
    collect_positive = not args.negatives_only
    collect_negatives = not args.positive_only
    
    collect_all_scores(
        triplets_json=Path(args.triplets),
        panels_dir=Path(args.panels_dir),
        output_json=Path(args.output),
        model=model,
        processor=processor,
        num_samples=args.num_samples,
        offset=args.offset,
        batch_size=args.batch_size,
        max_negatives=args.max_negatives,
        collect_positive=collect_positive,
        collect_negatives=collect_negatives
    )


if __name__ == '__main__':
    main()

