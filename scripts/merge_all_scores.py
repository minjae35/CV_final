"""
Merge all teacher score JSON files (positive + negatives)
"""
import json
import argparse
from pathlib import Path


def merge_all_scores(output_file: Path):
    """Merge all teacher score files"""
    
    input_files = [
        Path('results/teacher_scores_part1_1.json'),
        Path('results/teacher_scores_part1_2.json'),
        Path('results/teacher_scores_part2.json'),  # 기존 positive
        Path('results/teacher_scores_part2_neg.json')
    ]
    
    # Filter existing files
    existing_files = [f for f in input_files if f.exists()]
    
    if not existing_files:
        print("❌ No input files found!")
        return
    
    print(f"📂 Found {len(existing_files)} files to merge:")
    for f in existing_files:
        print(f"   - {f}")
    
    all_results = []
    total_failed = 0
    
    for input_file in existing_files:
        print(f"\nLoading {input_file}...")
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        failed = data.get('num_failed', 0)
        
        print(f"   ✅ {len(results)} samples ({failed} failed)")
        
        all_results.extend(results)
        total_failed += failed
    
    # Create merged output
    merged_data = {
        'num_samples': len(all_results),
        'num_failed': total_failed,
        'results': all_results
    }
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"\n✅ Merged {len(all_results)} scores (failed: {total_failed})")
    print(f"Results saved to: {output_file}")
    
    # Statistics
    positive_count = sum(1 for r in all_results if r.get('is_positive', True))
    negative_count = len(all_results) - positive_count
    
    print(f"\n📊 통계:")
    print(f"   - Positive samples: {positive_count}")
    print(f"   - Negative samples: {negative_count}")
    print(f"   - Total: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser(description='Merge All Teacher Scores')
    parser.add_argument('--output', type=str, default='results/teacher_scores_all.json')
    
    args = parser.parse_args()
    merge_all_scores(Path(args.output))


if __name__ == '__main__':
    main()

