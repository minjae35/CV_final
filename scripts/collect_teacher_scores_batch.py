"""
Collect Teacher Scores from Qwen2.5-VL with BATCH PROCESSING (2-3x faster)
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
  "score": 0.0-1.0
}
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


def query_teacher_model_batch(
    model,
    processor,
    panel_paths_batch: List[Dict[str, Path]],
    batch_size: int = 2
) -> List[Optional[Dict]]:
    """
    Query teacher model for multiple triplets at once
    
    Args:
        panel_paths_batch: List of dicts with keys 'A', 'B', 'C' (panel paths)
        batch_size: Number of triplets to process simultaneously
    
    Returns:
        List of analysis dicts (one per triplet)
    """
    if model is None:
        # Mock responses
        import random
        return [{
            "visual_continuity": "Mock",
            "temporal_flow": "Mock",
            "narrative_logic": "Mock",
            "overall_coherence": "Mock",
            "score": random.uniform(0.3, 0.9)
        } for _ in panel_paths_batch]
    
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
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,  # Enough for complete JSON
                do_sample=False,
                num_beams=1
            )
        
        # Decode each result
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        responses = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Parse each response (handle markdown code blocks)
        for response in responses:
            try:
                # Remove markdown code blocks if present
                if '```json' in response:
                    json_start = response.find('```json') + 7
                    json_end = response.find('```', json_start)
                    if json_end > json_start:
                        response = response[json_start:json_end].strip()
                elif '```' in response:
                    json_start = response.find('```') + 3
                    json_end = response.find('```', json_start)
                    if json_end > json_start:
                        response = response[json_start:json_end].strip()
                
                # Extract JSON
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                    results.append(result)
                else:
                    results.append(None)
            except:
                results.append(None)
        
        return results
    
    except Exception as e:
        print(f"❌ Batch error: {e}")
        return [None] * len(panel_paths_batch)


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


def collect_scores_batch(
    triplets_json: Path,
    panels_dir: Path,
    output_json: Path,
    model,
    processor,
    num_samples: Optional[int] = None,
    offset: int = 0,
    batch_size: int = 2,
    save_every: int = 25
):
    """Collect teacher scores with batch processing and periodic saving"""
    
    # Load triplets
    print(f"Loading triplets from {triplets_json}...")
    with open(triplets_json, 'r') as f:
        triplets = json.load(f)
    
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
            start_idx = len(results) + failed_count
        print(f"✅ Resuming from index {start_idx} ({len(results)} already collected)")
    
    triplets = triplets[start_idx:]
    print(f"Collecting scores for {len(triplets)} triplets (batch_size={batch_size})...")
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    # Process in batches
    for i in tqdm(range(0, len(triplets), batch_size), desc="Processing batches"):
        batch_triplets = triplets[i:i+batch_size]
        batch_paths = []
        batch_valid = []
        
        # Prepare batch
        for triplet in batch_triplets:
            panel_a_path = find_panel_image(panels_dir, triplet['A'])
            panel_b_path = find_panel_image(panels_dir, triplet['B'])
            panel_c_path = find_panel_image(panels_dir, triplet['C'])
            
            if all([panel_a_path, panel_b_path, panel_c_path]):
                batch_paths.append({
                    'A': panel_a_path,
                    'B': panel_b_path,
                    'C': panel_c_path
                })
                batch_valid.append(triplet)
            else:
                failed_count += 1
        
        if not batch_paths:
            continue
        
        # Query batch
        batch_results = query_teacher_model_batch(model, processor, batch_paths, batch_size)
        
        # Store results
        for triplet, analysis in zip(batch_valid, batch_results):
            if analysis:
                results.append({
                    'triplet': triplet,
                    'teacher_analysis': analysis,
                    'teacher_score': analysis.get('score', 0.5)
                })
            else:
                failed_count += 1
        
        # Periodic save
        batch_num = i // batch_size
        if (batch_num + 1) % save_every == 0:
            output_data = {
                'num_triplets': len(results),
                'num_failed': failed_count,
                'batch_size': batch_size,
                'results': results
            }
            with open(output_json, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 Checkpoint: {len(results)} scores saved")
    
    # Final save
    output_data = {
        'num_triplets': len(results),
        'num_failed': failed_count,
        'batch_size': batch_size,
        'results': results
    }
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Collected {len(results)} scores ({failed_count} failed)")
    print(f"Results saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description='Collect Teacher Scores (BATCH MODE)')
    parser.add_argument('--triplets', type=str,
                       default='data/processed_444_pages/triplets_444.json')
    parser.add_argument('--panels_dir', type=str,
                       default='data/processed_444_pages/cropped_panels')
    parser.add_argument('--output', type=str,
                       default='data/teacher_scores_batch.json')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--offset', type=int, default=0,
                       help='Starting index offset')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Number of triplets per batch (2-4 recommended)')
    parser.add_argument('--mock', action='store_true')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mock:
        print("Running in MOCK mode")
        model, processor = None, None
    else:
        model, processor = load_qwen_model(device)
    
    collect_scores_batch(
        triplets_json=Path(args.triplets),
        panels_dir=Path(args.panels_dir),
        output_json=Path(args.output),
        model=model,
        processor=processor,
        num_samples=args.num_samples,
        offset=args.offset,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()

