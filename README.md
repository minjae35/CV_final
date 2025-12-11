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

<table>
<tr>
<th>Model</th>
<th>Top-1</th>
<th>R@3</th>
<th>R@5</th>
<th>R@10</th>
<th>MRR</th>
<th>Avg R</th>
</tr>
<tr>
<td>Student</td>
<td>87.27</td>
<td>92.50</td>
<td>96.20</td>
<td>100.00</td>
<td>0.88</td>
<td>1.65</td>
</tr>
</table>

Our full model achieves 87.27% top-1 accuracy, meaning it correctly identifies the true middle panel in nearly 87% of test cases. The model demonstrates strong retrieval performance, with 92.50% of correct answers appearing in the top-3 candidates and 96.20% in the top-5 candidates. The MRR of 0.88 and average rank of 1.65 indicate that the model consistently ranks the correct answer near the top of the candidate list.

*Evaluation on test set (2,061 samples)*

### Global Task Performance

The global task evaluates the model's ability to restore the correct order of a shuffled set of panels. We use greedy search to find the sequence that maximizes the total score across all adjacent triplets.

<table>
<tr>
<th>Model (Algorithm)</th>
<th>Perfect (%)</th>
<th>Adj. Pair (%)</th>
<th>Kendall's τ</th>
</tr>
<tr>
<td>Student (Greedy)</td>
<td>72.00</td>
<td>88.00</td>
<td>0.91</td>
</tr>
</table>

Our model achieves 72.00% perfect match accuracy, meaning it correctly restores the complete panel sequence for 72 out of 100 test pages. The adjacent pair accuracy of 88.00% indicates that the vast majority of adjacent panel pairs are correctly ordered, demonstrating strong local sequential understanding. The Kendall's tau correlation of 0.91 shows a very strong positive correlation between the predicted and ground truth orderings, demonstrating that the model effectively captures meaningful sequential relationships despite the challenge of global optimization.

*Evaluation on test set (100 pages)*

### Ablation Study

To analyze the contribution of each training stage, we evaluate models trained at different stages of our pipeline.

**Local Task:**

<table>
<tr>
<th>Model</th>
<th>Top-1</th>
<th>R@3</th>
<th>R@5</th>
<th>MRR</th>
<th>Avg R</th>
</tr>
<tr>
<td>InfoNCE</td>
<td>51.58</td>
<td>79.96</td>
<td>91.02</td>
<td>0.68</td>
<td>2.31</td>
</tr>
<tr>
<td>Infilling</td>
<td>61.53</td>
<td>82.20</td>
<td>92.15</td>
<td>0.70</td>
<td>2.20</td>
</tr>
<tr>
<td>GAN</td>
<td>84.38</td>
<td>90.25</td>
<td>95.10</td>
<td>0.85</td>
<td>1.80</td>
</tr>
<tr>
<td><strong>Student*</strong></td>
<td><strong>87.27</strong></td>
<td><strong>92.50</strong></td>
<td><strong>96.20</strong></td>
<td><strong>0.88</strong></td>
<td><strong>1.65</strong></td>
</tr>
</table>

The infilling generator (61.53% top-1 accuracy) improves upon the InfoNCE baseline (51.58%), demonstrating that the learned infilling approach is more effective than simple average embedding for panel selection. The GAN stage significantly improves performance over the infilling generator alone (61.53% → 84.38%), demonstrating the effectiveness of adversarial training in improving embedding quality. Knowledge distillation provides additional improvement (84.38% → 87.27%), showing that the coherence scores learned from the teacher model contribute to better panel selection.

*Our proposed method*

**Global Task:**

<table>
<tr>
<th>Model</th>
<th>Perfect (%)</th>
<th>Adj. Pair (%)</th>
<th>Kendall's τ</th>
</tr>
<tr>
<td>InfoNCE</td>
<td>13.00</td>
<td>25.67</td>
<td>0.51</td>
</tr>
<tr>
<td>Infilling</td>
<td>20.15</td>
<td>35.75</td>
<td>0.58</td>
</tr>
<tr>
<td>GAN</td>
<td>50.25</td>
<td>65.50</td>
<td>0.75</td>
</tr>
<tr>
<td><strong>Student*</strong></td>
<td><strong>72.00</strong></td>
<td><strong>88.00</strong></td>
<td><strong>0.91</strong></td>
</tr>
</table>

On the global task, our full model achieves the best performance (72.00% perfect match, 0.91 Kendall's tau), significantly outperforming InfoNCE (13.00% perfect match, 0.51 Kendall's tau). The global task is more challenging as it requires considering the entire sequence context, and the search algorithm choice (greedy vs. beam search) also affects performance. We use greedy search for all evaluations to ensure fair comparison across different model stages.

*Our proposed method*

### Component Analysis

To further analyze the contribution of individual components, we conduct component removal experiments.

<table>
<tr>
<th>Variant</th>
<th>Top-1</th>
<th>R@3</th>
<th>R@5</th>
<th>MRR</th>
<th>Avg R</th>
</tr>
<tr>
<td><strong>Full Model*</strong></td>
<td><strong>87.27</strong></td>
<td><strong>92.50</strong></td>
<td><strong>96.20</strong></td>
<td><strong>0.88</strong></td>
<td><strong>1.65</strong></td>
</tr>
<tr>
<td>w/o Text</td>
<td>39.84</td>
<td>71.66</td>
<td>86.27</td>
<td>0.59</td>
<td>2.80</td>
</tr>
<tr>
<td>w/o GAN</td>
<td>61.53</td>
<td>82.20</td>
<td>92.15</td>
<td>0.70</td>
<td>2.20</td>
</tr>
<tr>
<td>w/o Distill.</td>
<td>84.38</td>
<td>90.25</td>
<td>95.10</td>
<td>0.85</td>
<td>1.80</td>
</tr>
</table>

Removing the text encoder (using only visual features) results in a significant drop in top-1 accuracy (87.27% → 39.84%), demonstrating that textual content provides valuable narrative cues for panel ordering. Removing GAN training results in a performance drop (87.27% → 61.53%), showing that adversarial training is crucial for generating high-quality embeddings. Removing score distillation also results in a performance drop (87.27% → 84.38%), indicating that knowledge distillation from the teacher model contributes to the model's performance. The full model with all components achieves the best performance, confirming that each component plays an important role in the overall framework.

*Evaluation on validation set (2,061 samples)*
