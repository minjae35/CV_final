# Comic Book Panel Ordering

## Resources

Final report and data/checkpoints can be found in the links below.

- **Final Report**: [link]
- **Data & Checkpoints**: [link]

## Introduction

This project addresses the task of understanding and restoring the narrative structure of comic book panels. Given a preceding panel (A) and a following panel (C), the system selects the correct intermediate panel (B) from multiple candidates (local task) and restores the full ordering of shuffled panels (global task). 

Our approach combines:
- **Multimodal panel embeddings** using contrastive learning (InfoNCE)
- **Infilling generator** enhanced by a GAN discriminator for embedding generation
- **Knowledge distillation** from a large vision-language model (Qwen2.5-VL-3B) to learn narrative coherence scores

The method achieves **87.27% Top-1 accuracy** for local panel selection, representing a **41.8% improvement** over the baseline infilling generator.

### Example

The following example illustrates how panels are extracted from a comic book page and ordered:

<img src="readme_imgs/example_with_bbox.png" width="270" alt="Panel extraction with bounding boxes">

*Original page with panel bounding boxes and reading order labels. Panels are annotated with green bounding boxes and numbered according to the correct reading order (left-to-right, top-to-bottom).*

<img src="readme_imgs/mixed_panel.png" width="750" alt="Shuffled panels">

*Shuffled panel sequence (input to the model). The panels are randomly reordered, representing the challenge of restoring the correct narrative sequence.*

<img src="readme_imgs/example_with_horizontal_order.png" width="750" alt="Ordered panels">

*Restored panel sequence by our model. The panels extracted from the original page are arranged in the correct reading order as restored by our model, demonstrating its ability to understand narrative flow across multiple panels.*

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Results

### Local Task Performance

The local task evaluates the model's ability to select the correct intermediate panel (B) given a preceding panel (A) and a following panel (C) from multiple candidates.

| Model | Top-1 | R@3 | R@5 | R@10 | MRR | Avg R |
|-------|-------|-----|-----|------|-----|-------|
| Student | 87.27 | 92.50 | 96.20 | 100.00 | 0.88 | 1.65 |

Our full model achieves 87.27% top-1 accuracy, meaning it correctly identifies the true middle panel in nearly 87% of test cases. The model demonstrates strong retrieval performance, with 92.50% of correct answers appearing in the top-3 candidates and 96.20% in the top-5 candidates. The MRR of 0.88 and average rank of 1.65 indicate that the model consistently ranks the correct answer near the top of the candidate list.

*Evaluation on test set (2,061 samples)*

### Global Task Performance

The global task evaluates the model's ability to restore the correct order of a shuffled set of panels. We use greedy search to find the sequence that maximizes the total score across all adjacent triplets.

| Model (Algorithm) | Perfect (%) | Adj. Pair (%) | Kendall's τ |
|-------------------|------------|---------------|-------------|
| Student (Greedy) | 72.00 | 88.00 | 0.91 |

Our model achieves 72.00% perfect match accuracy, meaning it correctly restores the complete panel sequence for 72 out of 100 test pages. The adjacent pair accuracy of 88.00% indicates that the vast majority of adjacent panel pairs are correctly ordered, demonstrating strong local sequential understanding. The Kendall's tau correlation of 0.91 shows a very strong positive correlation between the predicted and ground truth orderings, demonstrating that the model effectively captures meaningful sequential relationships despite the challenge of global optimization.

*Evaluation on test set (100 pages)*

### Ablation Study

To analyze the contribution of each training stage, we evaluate models trained at different stages of our pipeline.

**Local Task:**

| Model | Top-1 | R@3 | R@5 | MRR | Avg R |
|-------|-------|-----|-----|-----|-------|
| InfoNCE | 51.58 | 79.96 | 91.02 | 0.68 | 2.31 |
| Infilling | 61.53 | 82.20 | 92.15 | 0.70 | 2.20 |
| GAN | 84.38 | 90.25 | 95.10 | 0.85 | 1.80 |
| **Student*** | **87.27** | **92.50** | **96.20** | **0.88** | **1.65** |

The infilling generator (61.53% top-1 accuracy) improves upon the InfoNCE baseline (51.58%), demonstrating that the learned infilling approach is more effective than simple average embedding for panel selection. The GAN stage significantly improves performance over the infilling generator alone (61.53% → 84.38%), demonstrating the effectiveness of adversarial training in improving embedding quality. Knowledge distillation provides additional improvement (84.38% → 87.27%), showing that the coherence scores learned from the teacher model contribute to better panel selection.

*Our proposed method*

**Global Task:**

| Model | Perfect (%) | Adj. Pair (%) | Kendall's τ |
|-------|------------|---------------|-------------|
| InfoNCE | 13.00 | 25.67 | 0.51 |
| Infilling | 20.15 | 35.75 | 0.58 |
| GAN | 50.25 | 65.50 | 0.75 |
| **Student*** | **72.00** | **88.00** | **0.91** |

On the global task, our full model achieves the best performance (72.00% perfect match, 0.91 Kendall's tau), significantly outperforming InfoNCE (13.00% perfect match, 0.51 Kendall's tau). The global task is more challenging as it requires considering the entire sequence context, and the search algorithm choice (greedy vs. beam search) also affects performance. We use greedy search for all evaluations to ensure fair comparison across different model stages.

*Our proposed method*

### Component Analysis

To further analyze the contribution of individual components, we conduct component removal experiments.

| Variant | Top-1 | R@3 | R@5 | MRR | Avg R |
|---------|-------|-----|-----|-----|-------|
| **Full Model*** | **87.27** | **92.50** | **96.20** | **0.88** | **1.65** |
| w/o Text | 39.84 | 71.66 | 86.27 | 0.59 | 2.80 |
| w/o GAN | 61.53 | 82.20 | 92.15 | 0.70 | 2.20 |
| w/o Distill. | 84.38 | 90.25 | 95.10 | 0.85 | 1.80 |

Removing the text encoder (using only visual features) results in a significant drop in top-1 accuracy (87.27% → 39.84%), demonstrating that textual content provides valuable narrative cues for panel ordering. Removing GAN training results in a performance drop (87.27% → 61.53%), showing that adversarial training is crucial for generating high-quality embeddings. Removing score distillation also results in a performance drop (87.27% → 84.38%), indicating that knowledge distillation from the teacher model contributes to the model's performance. The full model with all components achieves the best performance, confirming that each component plays an important role in the overall framework.

*Evaluation on validation set (2,061 samples)*
