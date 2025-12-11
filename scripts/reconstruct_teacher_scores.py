"""
Reconstruct teacher scores for training:
- Merge all score files
- For negative samples, replace triplet['C'] with panel_C_used
- Create unified dataset with both positive and negative samples
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


def reconstruct_negative_triplet(result: Dict) -> Dict:
    """Reconstruct negative triplet: replace C with panel_C_used"""
    if result.get('is_positive', True):
        return result['triplet']  # No change for positive
    
    # For negative, use panel_C_used as the actual C panel
    triplet = result['triplet'].copy()
    if 'panel_C_used' in result:
        triplet['C'] = result['panel_C_used']
    else:
        # Fallback: if panel_C_used not found, keep original C
        print(f"⚠️  Warning: panel_C_used not found for negative sample, using original C")
    
    return triplet


def load_and_reconstruct(input_files: List[Path]) -> List[Dict]:
    """Load all files and reconstruct triplets"""
    all_results = []
    
    for input_file in input_files:
        if not input_file.exists():
            print(f"⚠️  Skipping {input_file} (not found)")
            continue
        
        print(f"\n📂 Loading {input_file.name}...")
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        print(f"   Found {len(results)} samples")
        
        # Reconstruct each result
        for result in tqdm(results, desc=f"   Reconstructing {input_file.name}"):
            # Reconstruct triplet for negative samples
            reconstructed_triplet = reconstruct_negative_triplet(result)
            
            # Create new result entry
            new_result = {
                'triplet_index': result.get('triplet_index', -1),
                'is_positive': result.get('is_positive', True),
                'triplet': reconstructed_triplet,
                'teacher_analysis': result.get('teacher_analysis', {}),
                'teacher_score': result.get('teacher_score', 0.5)
            }
            
            # Keep panel_C_used for reference (even for positive)
            if 'panel_C_used' in result:
                new_result['panel_C_used'] = result['panel_C_used']
            
            all_results.append(new_result)
    
    return all_results


def save_reconstructed(output_file: Path, all_results: List[Dict]):
    """Save reconstructed results"""
    # Statistics
    positive_count = sum(1 for r in all_results if r.get('is_positive', True))
    negative_count = len(all_results) - positive_count
    
    merged_data = {
        'num_samples': len(all_results),
        'num_positive': positive_count,
        'num_negative': negative_count,
        'results': all_results
    }
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"\n✅ Saved {len(all_results)} reconstructed scores to {output_file}")
    print(f"   - Positive: {positive_count}")
    print(f"   - Negative: {negative_count}")


def verify_reconstruction(results: List[Dict], num_samples: int = 5):
    """Verify that negative triplets are correctly reconstructed"""
    print(f"\n🔍 Verifying reconstruction (checking {num_samples} samples)...")
    
    negative_samples = [r for r in results if not r.get('is_positive', True)][:num_samples]
    
    for i, result in enumerate(negative_samples):
        triplet = result['triplet']
        panel_C_used = result.get('panel_C_used', {})
        
        print(f"\n  Sample {i+1} (negative):")
        print(f"    Triplet C: {triplet['C'].get('panel_index', 'N/A')}")
        print(f"    panel_C_used: {panel_C_used.get('panel_index', 'N/A')}")
        
        # Check if they match (they should match after reconstruction)
        if triplet['C'].get('panel_index') == panel_C_used.get('panel_index'):
            print(f"    ✅ Correctly reconstructed")
        else:
            print(f"    ⚠️  Mismatch detected!")


def main():
    parser = argparse.ArgumentParser(description='Reconstruct Teacher Scores for Training')
    parser.add_argument('--output', type=str, 
                       default='results/teacher_scores_reconstructed.json',
                       help='Output file path')
    parser.add_argument('--verify', action='store_true',
                       help='Verify reconstruction')
    
    args = parser.parse_args()
    
    # Input files (in order)
    input_files = [
        Path('results/teacher_scores_part1_1.json'),
        Path('results/teacher_scores_part1_2.json'),
        Path('results/teacher_scores_part2.json'),
        Path('results/teacher_scores_part2_neg.json')
    ]
    
    print("🔄 Reconstructing teacher scores for training...")
    print(f"📁 Input files: {len(input_files)}")
    
    # Load and reconstruct
    all_results = load_and_reconstruct(input_files)
    
    # Verify if requested
    if args.verify:
        verify_reconstruction(all_results)
    
    # Save
    save_reconstructed(Path(args.output), all_results)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()





