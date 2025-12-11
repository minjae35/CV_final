#!/bin/bash

# Optimized parallel collection
# - Negative: 1 per triplet
# - Parallel: 2 processes
# - batch_size: 2
# - Estimated time: ~2.5 hours

cd /home/mjb4835/CV_final_project
source /opt/conda/etc/profile.d/conda.sh
conda activate CV_final

echo "🚀 Starting safe parallel collection"
echo "📊 Configuration:"
echo "   - Negative: 1 per triplet"
echo "   - Parallel: 2 processes (memory safe)"
echo "   - Batch size: 2 (memory safe)"
echo "📊 Estimated time: ~2.5 hours"
echo ""

# Part 1-1: triplets 0-700 (positive + negatives)
python scripts/collect_all_scores.py \
    --num_samples 700 \
    --batch_size 2 \
    --max_negatives 1 \
    --panels_dir data/processed_444_pages/cropped_panels \
    --output results/teacher_scores_part1_1.json &

PID1=$!
echo "✅ Part 1-1 started (PID: $PID1) - Triplets 0-699 (pos+neg)"

# Part 1-2: triplets 700-1050 (positive + negatives)
sleep 15
python scripts/collect_all_scores.py \
    --offset 700 \
    --num_samples 350 \
    --batch_size 2 \
    --max_negatives 1 \
    --panels_dir data/processed_444_pages/cropped_panels \
    --output results/teacher_scores_part1_2.json &

PID2=$!
echo "✅ Part 1-2 started (PID: $PID2) - Triplets 700-1049 (pos+neg)"

echo ""
echo "⏳ Part 1 processes running..."
echo "   Progress bars can be monitored in real-time below"
echo ""

# Wait for Part 1 processes
wait $PID1
echo ""
echo "✅ Part 1-1 completed"

wait $PID2
echo "✅ Part 1-2 completed"

echo ""
echo "✅ Part 1 completed! Starting Part 2..."

python scripts/collect_all_scores.py \
    --offset 1050 \
    --num_samples 1044 \
    --batch_size 2 \
    --max_negatives 1 \
    --negatives_only \
    --panels_dir data/processed_444_pages/cropped_panels \
    --output results/teacher_scores_part2_neg.json

echo "✅ Part 2 completed"

echo ""
echo "🎉 All collection completed!"
echo ""
echo "📦 Collected files:"
ls -lh results/teacher_scores_part*.json
echo ""
echo "💡 Next step: python scripts/merge_all_scores.py"

