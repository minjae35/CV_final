"""
Merge multiple teacher score JSON files
"""
import json
import argparse
from pathlib import Path


def merge_scores(input_files, output_file):
    """Merge teacher scores from multiple JSON files"""
    
    all_results = []
    total_failed = 0
    
    for input_file in input_files:
        print(f"Loading {input_file}...")
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        all_results.extend(data['results'])
        total_failed += data.get('num_failed', 0)
    
    # Create merged output
    merged_data = {
        'num_triplets': len(all_results),
        'num_failed': total_failed,
        'results': all_results
    }
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"\n✅ Merged {len(all_results)} scores")
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Merge Teacher Scores')
    parser.add_argument('--input', type=str, nargs='+', help='Input JSON files')
    parser.add_argument('--output', type=str, default='data/teacher_scores_all.json')
    
    args = parser.parse_args()
    
    # If no inputs provided, use default pattern
    if not args.input:
        input_files = [
            Path('data/teacher_scores_part1.json'),
            Path('data/teacher_scores_part2.json'),
            Path('data/teacher_scores_part3.json')
        ]
        # Filter existing files only
        input_files = [f for f in input_files if f.exists()]
    else:
        input_files = [Path(f) for f in args.input]
    
    if not input_files:
        print("❌ No input files found!")
        return
    
    merge_scores(input_files, Path(args.output))


if __name__ == '__main__':
    main()

