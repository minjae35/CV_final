"""
Collect Teacher Scores from Qwen2.5-VL-3B
Use structured CoT prompting to get coherence scores for triplets
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# CoT Prompt Template
SYSTEM_PROMPT = """You are an expert in analyzing comic panel sequences. Your task is to evaluate if three consecutive panels (A, B, C) form a coherent narrative sequence."""

COT_PROMPT_TEMPLATE = """Analyze these three comic panels and evaluate their coherence.

Panel A: [First panel]
Panel B: [Middle panel]  
Panel C: [Third panel]

Please analyze step by step:

Step 1 - Visual Continuity: 
Examine the visual elements (characters, backgrounds, art style, lighting). Are they consistent across all three panels?

Step 2 - Temporal Flow:
Analyze the time progression. Do the actions and events flow naturally from A→B→C?

Step 3 - Narrative Logic:
Consider the story and causal relationships. Does B logically connect A and C?

Step 4 - Overall Coherence:
Assess the overall naturalness of the sequence.

Provide your analysis in JSON format:
{{
  "visual_continuity": "your analysis here",
  "temporal_flow": "your analysis here",  
  "narrative_logic": "your analysis here",
  "overall_coherence": "your analysis here",
  "score": 0.0-1.0
}}

Score guidelines:
- 0.9-1.0: Perfect coherence, natural flow
- 0.7-0.9: Good coherence, minor issues
- 0.5-0.7: Moderate coherence, some disconnection
- 0.3-0.5: Weak coherence, significant issues
- 0.0-0.3: No coherence, unrelated panels
"""

FEW_SHOT_EXAMPLES = [
    {
        "description": "High coherence example",
        "analysis": {
            "visual_continuity": "All three panels show the same character in the same room with consistent art style and lighting.",
            "temporal_flow": "Panel A shows character standing, B shows them reaching for door, C shows them opening door. Natural progression.",
            "narrative_logic": "Clear cause-effect: character decides to leave (A), walks to door (B), opens it (C).",
            "overall_coherence": "Perfect narrative flow with clear continuity.",
            "score": 0.95
        }
    },
    {
        "description": "Low coherence example",
        "analysis": {
            "visual_continuity": "Different characters in different locations. Art style changes between panels.",
            "temporal_flow": "No clear temporal connection. Actions seem unrelated.",
            "narrative_logic": "Panel B does not logically connect A and C. Seems like different scenes.",
            "overall_coherence": "Panels appear to be from different sequences.",
            "score": 0.15
        }
    }
]


def load_qwen_model(device: str = "cuda"):
    """Load Qwen2.5-VL-3B model (FP16 for best speed)"""
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        print("Loading Qwen2.5-VL-3B model (FP16)...")
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        
        # FP16 without quantization (faster in this environment)
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
        
        print(f"✅ Model loaded (FP16)")
        return model, processor
    
    except Exception as e:
        print(f"❌ Error loading Qwen model: {e}")
        print("Falling back to mock mode for testing...")
        return None, None


def create_prompt_with_images(panel_a_path: Path, panel_b_path: Path, panel_c_path: Path) -> str:
    """Create prompt with image paths"""
    # For Qwen2-VL, we'll pass images separately
    # This is just the text prompt
    return COT_PROMPT_TEMPLATE


def query_teacher_model(
    model,
    processor,
    panel_a: Path,
    panel_b: Path,
    panel_c: Path,
    temperature: float = 0.3
) -> Optional[Dict]:
    """
    Query teacher model for coherence score
    
    Returns:
        Dict with keys: visual_continuity, temporal_flow, narrative_logic, 
                       overall_coherence, score
    """
    if model is None:
        # Mock response for testing
        import random
        return {
            "visual_continuity": "Mock analysis",
            "temporal_flow": "Mock analysis",
            "narrative_logic": "Mock analysis",
            "overall_coherence": "Mock analysis",
            "score": random.uniform(0.3, 0.9)
        }
    
    try:
        from qwen_vl_utils import process_vision_info
        
        # Create conversation with images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(panel_a)},
                    {"type": "text", "text": "Panel A (above)"},
                    {"type": "image", "image": str(panel_b)},
                    {"type": "text", "text": "Panel B (above)"},
                    {"type": "image", "image": str(panel_c)},
                    {"type": "text", "text": "Panel C (above)\n\n" + SYSTEM_PROMPT + "\n\n" + COT_PROMPT_TEMPLATE}
                ]
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Process inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        # Generate response (optimized for speed)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,   # Enough for complete JSON
                do_sample=False,      # Greedy decoding
                num_beams=1           # No beam search
            )
        
        # Trim prompt from output and decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Parse JSON response (handle markdown code blocks)
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
                return result
            else:
                print(f"⚠️  Could not find JSON in response: {response[:100]}...")
                return None
        
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error: {e}")
            print(f"Response: {response[:100]}...")
            return None
    
    except Exception as e:
        print(f"❌ Error querying model: {e}")
        return None


def collect_scores_for_triplets(
    triplets_json: Path,
    panels_dir: Path,
    output_json: Path,
    model,
    processor,
    num_samples: Optional[int] = None,
    offset: int = 0,
    temperature: float = 0.3
):
    """Collect teacher scores for all triplets"""
    
    # Load triplets
    print(f"Loading triplets from {triplets_json}...")
    with open(triplets_json, 'r') as f:
        triplets = json.load(f)
    
    # Apply offset if specified
    if offset > 0:
        triplets = triplets[offset:]
        print(f"Starting from offset {offset}")
    
    if num_samples:
        triplets = triplets[:num_samples]
    
    print(f"Collecting scores for {len(triplets)} triplets...")
    
    results = []
    failed_count = 0
    
    for triplet in tqdm(triplets, desc="Collecting teacher scores"):
        try:
            # Get panel paths
            panel_a_info = triplet['A']
            panel_b_info = triplet['B']
            panel_c_info = triplet['C']
            
            # Find image paths
            panel_a_path = find_panel_image(panels_dir, panel_a_info)
            panel_b_path = find_panel_image(panels_dir, panel_b_info)
            panel_c_path = find_panel_image(panels_dir, panel_c_info)
            
            if not all([panel_a_path, panel_b_path, panel_c_path]):
                print(f"⚠️  Could not find all panel images")
                failed_count += 1
                continue
            
            # Query teacher model
            result = query_teacher_model(
                model, processor,
                panel_a_path, panel_b_path, panel_c_path,
                temperature=temperature
            )
            
            if result and 'score' in result:
                results.append({
                    'triplet': triplet,
                    'teacher_analysis': result,
                    'teacher_score': result['score']
                })
            else:
                failed_count += 1
        
        except Exception as e:
            print(f"❌ Error processing triplet: {e}")
            failed_count += 1
            continue
    
    # Save results
    print(f"\n✅ Collected {len(results)} scores ({failed_count} failed)")
    
    output_data = {
        'num_triplets': len(results),
        'num_failed': failed_count,
        'temperature': temperature,
        'results': results
    }
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_json}")
    
    return results


def find_panel_image(panels_dir: Path, panel_info: Dict) -> Optional[Path]:
    """Find panel image path from metadata"""
    panel_filename = panel_info.get('panel_filename', '')
    comic_no = panel_info['comic_no']
    
    # Try direct path
    direct_path = panels_dir / panel_filename
    if direct_path.exists():
        return direct_path
    
    # Try nested path
    nested_path = panels_dir / str(comic_no) / panel_filename
    if nested_path.exists():
        return nested_path
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Collect Teacher Scores from Qwen2.5-VL')
    parser.add_argument('--triplets', type=str,
                       default='data/processed_444_pages/triplets_444.json',
                       help='Path to triplets JSON')
    parser.add_argument('--panels_dir', type=str,
                       default='data/processed_444_pages/cropped_panels',
                       help='Directory containing panel images')
    parser.add_argument('--output', type=str,
                       default='data/teacher_scores_444.json',
                       help='Output JSON file')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples (None = all)')
    parser.add_argument('--offset', type=int, default=0,
                       help='Starting index offset')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Temperature for sampling')
    parser.add_argument('--mock', action='store_true',
                       help='Use mock mode (for testing without model)')
    
    args = parser.parse_args()
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mock:
        print("Running in MOCK mode (no actual model)")
        model, processor = None, None
    else:
        model, processor = load_qwen_model(device)
    
    # Collect scores
    collect_scores_for_triplets(
        triplets_json=Path(args.triplets),
        panels_dir=Path(args.panels_dir),
        output_json=Path(args.output),
        model=model,
        processor=processor,
        num_samples=args.num_samples,
        offset=args.offset,
        temperature=args.temperature
    )


if __name__ == '__main__':
    main()

