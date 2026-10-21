# Caltech-UCSD Birds 200 (CUB-200-2011) Classification System

**Course:** COS30082 Applied Machine Learning - Assignment 1  
**Author:** Sharon Lee  
**Repository:** [https://github.com/jysharon-lee/COS30082---Applied-Machine-Learning](https://github.com/jysharon-lee/COS30082---Applied-Machine-Learning)

---

## Project Overview

This repository contains the end-to-end deep learning pipeline, experimental evaluations, and interactive deployment for fine-grained bird species classification on the **Caltech-UCSD Birds 200 (CUB-200-2011)** benchmark. The task requires categorizing images across 200 bird species with extreme visual similarity and subtle inter-class variations, under a constrained dataset size (4,829 training images, 1,204 test images; averaging ~24 training images per class).

To address these challenges, we systematically benchmark three distinct architectural paradigms:
1. **ResNet-101** (Classical Deep Residual CNN)
2. **EfficientNet-B2** (Compound-Scaled CNN with MBConv blocks)
3. **Swin-T** (Hierarchical Vision Transformer with Shifted Window Attention) — **Champion Model**

The project includes advanced training schemes (Layer-Wise Learning Rate Decay, Label Smoothing, Cosine Warmup Annealing), inference-time ensembling (Test-Time Augmentation), Explainable AI (Grad-CAM saliency mapping), and a production-grade Gradio web application with a multi-model architecture comparator.

---

## Live Deployments & Media

- **Interactive Web Application:** `https://[your-active-link].gradio.live` *(or run locally via `python src/app.py`)*
- **Video Presentation (10 Minutes):** `[Your YouTube Presentation URL]`
- **Official 4-Page Report:** Available in `docs/` or submitted via Canvas.

---

## Benchmark Results Summary

All models were evaluated under identical data partitions (85% train / 15% stratified validation; 1,204 independent test samples) using ImageNet-pretrained feature representations.

| Model Architecture | Model Family | Parameters | Resolution | Test Top-1 Acc. | Test Per-Class Acc. ($Ave$) | Test Top-5 Acc. | Notes / Key Findings |
|---|---|---|---|---|---|---|---|
| **ResNet-101** | Deep Residual CNN | 44.5 M | 224 px | 84.50% | 84.10% | 95.50% | Strong baseline; heavy parameter cost |
| **EfficientNet-B2** | Compound-Scaled CNN | 8.0 M | 224 px | 82.50% | — | — | Uniform resolution baseline |
| **EfficientNet-B2 (Native)** | Compound-Scaled CNN | **8.0 M** | 260 px | 82.06% | 81.59% | 95.02% | **Most parameter-efficient** (5.5x smaller than ResNet) |
| **Swin-T (LLRD)** | Hierarchical ViT | 27.7 M | 224 px | 83.89% | — | — | Layer-Wise LR Decay ($\gamma = 0.85$) |
| **Swin-T (LLRD + TTA)** | Hierarchical ViT + TTA | 27.7 M | 224 px | 84.80% | — | — | **+0.91% gain** from Test-Time Augmentation |
| **Swin-T (Champion)** | **Hierarchical ViT** | **27.7 M** | **224 px** | **86.05%** | **85.78%** | **96.10%** | **Champion Model** (highest accuracy across all metrics) |

*Metric Formula: Average Per-Class Accuracy is computed as $Ave = \frac{1}{C}\sum_{i=1}^C T_i$, where $C = 200$ and $T_i$ is the accuracy for class $i$.*

---

## Key Methodological Innovations

1. **Shifted Window Self-Attention (Swin-T):**  
   Captures long-range spatial correlations between disjoint anatomical features (e.g. concurrent beak curvature, crown crest, and tail pattern) in a single forward pass, overcoming the locality constraints of convolution.
2. **Layer-Wise Learning Rate Decay (LLRD):**  
   Implements geometric learning rate decay ($\gamma = 0.85$) from deeper stages down to shallow patch-embedding layers ($\eta_l = \eta_{\text{base}} \cdot \gamma^{L-l}$), preserving general visual filters while adapting task-specific semantic layers.
3. **Label Smoothing Regularization ($\epsilon = 0.1$):**  
   Softens hard one-hot targets to penalize overconfident output distributions, mitigating overfitting on small per-class training splits.
4. **Test-Time Augmentation (TTA):**  
   Ensembles predictions across original and horizontally flipped test samples at inference time, yielding up to a **+0.91%** accuracy improvement with zero added training cost.
5. **Explainable AI (Grad-CAM):**  
   Computes gradient-weighted class activation maps at Stage 4 of Swin-T, visually verifying that classifications are driven by authentic biological morphology rather than background image context.

---

## Repository Structure

```text
COS30082---Applied-Machine-Learning/
├── AML Assignment 1/              # Official CUB-200 Dataset files
│   ├── Train/                     # 4,829 training images
│   ├── Test/                      # 1,204 test images
│   ├── train.txt                  # Image names and class indices
│   └── test.txt                   # Test split definitions
├── notebooks/
│   └── colab_run.ipynb            # Master Google Colab training & plotting notebook
├── src/
│   ├── __init__.py
│   ├── app.py                     # Interactive Gradio web application with Grad-CAM & comparator
│   ├── classes.json               # Ordered list of all 200 bird species
│   ├── config.py                  # Global hyperparameters, device setup, and paths
│   ├── data.py                    # DataLoaders, augmentation pipelines, and stratified splits
│   ├── models.py                  # Model registry (ResNet-101, EfficientNet-B2, Swin-T)
│   ├── train.py                   # Training loop, AdamW, LLRD, AMP, TTA, and evaluation
│   └── swin_t_best.pth            # Champion model checkpoint weights
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation and guide
```

---

## Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/jysharon-lee/COS30082---Applied-Machine-Learning.git
cd COS30082---Applied-Machine-Learning
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision gradio
```

---

## Running the Web Application

To launch the interactive Gradio web application locally:

```bash
python src/app.py
```

- **Running on public URL:** In your terminal, open [http://something-something.gradio.live] in your web browser.

### Features of the Web Application:
- **Interactive Classification:** Upload any image or test sample to receive instant predictions with latency metrics.
- **Top-5 Probability Distribution:** Visualizes calibrated confidence scores across candidate classes.
- **Explainable AI (XAI):** Renders Stage-4 Grad-CAM saliency heatmaps highlighting anatomical bird regions (beak, crest, wing bars).
- **Multi-Model Comparator:** Toggle between Swin-T, ResNet-101, and EfficientNet-B2 to compare model decisions side-by-side.

---

## Training on Google Colab

The complete experiment pipeline is packaged in `notebooks/colab_run.ipynb`.

1. Open `notebooks/colab_run.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Execute Sections 1 through 5 to mount Google Drive and install requirements.
4. Run Section 8 to train the baselines or Section 9 to run TTA evaluation.
5. Execute Section 10 to generate publication-grade 300 DPI figures and the final results CSV summary table:
   - `figure1_architecture_comparison.png`
   - `figure2_model_comparison_bars.png`
   - `figure3_champion_swin_t_learning_curves.png`
   - `figure4_confusion_matrix_heatmap.png`
   - `figure5_top10_confusion_pairs.png`
   - `final_models_summary.csv`

---
