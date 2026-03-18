<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <a href="https://www.kaggle.com/datasets/emirsecer/vegetables"><img src="https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle Dataset"></a>
</p>

<h1 align="center">🥬 Vegetable Image Classification<br><sub>From Classical ML to Ensemble — A Comprehensive Comparative Analysis</sub></h1>

<p align="center">
  <strong>Undergraduate Thesis</strong> &nbsp;·&nbsp;
  23 classes &nbsp;·&nbsp; 6,170 images &nbsp;·&nbsp; 30+ models &nbsp;·&nbsp; 100% accuracy
</p>

---

## 📖 Abstract

This study presents a **comprehensive image classification research** on **6,170 vegetable images across 23 categories**, spanning classical machine learning, deep learning, Vision Transformers, and ensemble methods.

| Approach | Best Accuracy | Top Model |
|:---------|:------------:|:----------|
| Classical ML | 63.33% | Random Forest |
| Custom CNN | 92.98% | Depthwise Separable CNN |
| Transfer Learning | **100.00%** | EfficientNetV2-S · ConvNeXt-Tiny |
| Vision Transformers | **100.00%** | ViT-Small/16 · CoAtNet-0 · EfficientFormer-L1 |
| Self-Supervised (SimCLR) | 90.76% | ResNet-18 (no labels) |
| Ensemble | **100.00%** | Hard/Soft Voting · Stacking (XGBoost) |

> **Key finding:** Transfer learning and transformer-based models achieved **~37 percentage points** higher accuracy compared to classical ML with handcrafted features. Ensemble strategies reached perfect agreement with Cohen's Kappa = 1.0.

---

## 🗂️ Project Structure

```
vegetable-classification-thesis/
│
├── 📓 01_eda_feature_engineering/     ← Exploratory data analysis & 500+ feature extraction
│   ├── 01_eda_feature_engineering.ipynb
│   └── results/                       (9 visualizations)
│
├── 📓 02_classical_ml/                ← 10 classical ML models & Optuna optimization
│   ├── 02_classical_ml.ipynb
│   └── results/                       (4 visualizations + CSV)
│
├── 📓 03_cnn_transfer_learning/       ← Custom CNNs + 4 transfer learning models
│   ├── 03_cnn_transfer_learning.ipynb
│   └── results/                       (2 visualizations)
│
├── 📓 04_vision_transformers/         ← ViT, Swin, DeiT, CoAtNet, EfficientFormer
│   ├── 04_vision_transformers.ipynb
│   └── results/                       (2 visualizations)
│
├── 📓 05_advanced_techniques/         ← SimCLR, Metric Learning, Focal Loss, NAS
│   ├── 05_advanced_techniques.ipynb
│   └── results/                       (6 visualizations)
│
├── 📓 06_ensemble/                    ← 8 ensemble strategies & statistical tests
│   ├── 06_ensemble.ipynb
│   └── results/                       (4 visualizations)
│
├── 📄 report/
│   ├── main.tex                       ← LaTeX thesis report
│   ├── main.pdf                       ← Compiled PDF
│   ├── main.docx                      ← Word version
│   └── references.bib                 ← 30 academic references
│
├── 📋 requirements.txt
└── 📘 README.md
```

---

## 📊 Dataset

| Property | Value |
|:---------|:------|
| **Source** | [Kaggle — Vegetables](https://www.kaggle.com/datasets/emirsecer/vegetables) |
| **Total images** | 6,170 |
| **Number of classes** | 23 |
| **Input size** | 224 × 224 px |
| **Train / Validation / Test** | 70% / 15% / 15% (stratified) |
| **Problem type** | Multi-class image classification |

**Classes:** Bean · Bitter Gourd · Bottle Gourd · Brinjal · Broccoli · Cabbage · Capsicum · Carrot · Cauliflower · Cucumber · Papaya · Potato · Pumpkin · Radish · Tomato and others.

---

## 📓 Notebook Details

### 1 · Exploratory Data Analysis & Feature Engineering

> `01_eda_feature_engineering/01_eda_feature_engineering.ipynb`

In-depth analysis of the dataset with **500+ handcrafted features** extracted.

| Feature Group | Technique | Dimensions |
|:-------------|:----------|----------:|
| Color histogram | HSV + LAB (32 bins × 3 channels × 2) | 192 |
| Texture | LBP (uniform) + GLCM (5 properties × 4 angles) | 30 |
| Shape | HOG (9 orientations, 8×8 cells) + Hu Moments | 191 |
| Edge | Canny density + Sobel (mean & std) | 3 |
| Statistical | RGB (mean, std, skew, kurtosis) + HSV (mean, std) | 18 |
| Frequency | FFT energy, entropy, low/high frequency ratio | 4 |

**Visualizations:** Class distribution · RGB histograms · Quality metrics · t-SNE · UMAP · PCA · Inter-class similarity matrix

<details>
<summary>📸 Sample visualizations</summary>

| | |
|:---:|:---:|
| ![Class Distribution](01_eda_feature_engineering/results/class_distribution.png) | ![RGB Histograms](01_eda_feature_engineering/results/rgb_histograms.png) |
| ![t-SNE](01_eda_feature_engineering/results/tsne_visualization.png) | ![UMAP](01_eda_feature_engineering/results/umap_visualization.png) |

</details>

---

### 2 · Classical Machine Learning

> `02_classical_ml/02_classical_ml.ipynb`

**10 different ML models** trained on handcrafted features with Optuna hyperparameter optimization.

| Model | Accuracy | Macro F1 | CV Mean | Config |
|:------|:--------:|:--------:|:-------:|:-------|
| **Random Forest** | **0.6333** | **0.4519** | 0.4071 | 200 trees |
| SVM-Linear | 0.5000 | 0.3102 | 0.3286 | C=1.0 |
| LightGBM | 0.4667 | 0.2961 | 0.2214 | 200 estimators |
| Gaussian NB | 0.4333 | 0.2602 | 0.1571 | — |
| Logistic Regression | 0.4333 | 0.2809 | 0.3500 | OVR |
| SVM-RBF | 0.4000 | 0.2308 | 0.1786 | C=0.1 |
| XGBoost | 0.3667 | 0.1981 | 0.3000 | Optuna |
| KNN (k=5) | 0.3333 | 0.1684 | 0.2500 | — |
| CatBoost | 0.3333 | 0.1555 | 0.2786 | 200 iterations |
| SVM-Polynomial | 0.1667 | 0.0159 | 0.1429 | degree=3 |

**Additional techniques:** PCA dimensionality reduction (95% variance → 40 components) · LDA · Stratified 5-Fold CV · GridSearchCV

<details>
<summary>📸 Result visualizations</summary>

| | |
|:---:|:---:|
| ![Model Comparison](02_classical_ml/results/model_comparison_chart.png) | ![Confusion Matrix](02_classical_ml/results/confusion_matrix_Random_Forest.png) |
| ![PCA Analysis](02_classical_ml/results/pca_analysis.png) | ![Feature Importance](02_classical_ml/results/rf_feature_importance.png) |

</details>

---

### 3 · CNN & Transfer Learning

> `03_cnn_transfer_learning/03_cnn_transfer_learning.ipynb`

#### Custom CNN Architectures

| Model | Test Accuracy | Macro F1 | Parameters |
|:------|:------------:|:--------:|:---------:|
| SimpleCNN | 0.8855 | 0.8548 | 2.5M |
| ResidualCNN | 0.8801 | 0.8376 | 2.1M |
| Depthwise Separable CNN | 0.9298 | 0.9173 | **0.2M** |

#### Transfer Learning

| Model | Test Accuracy | Macro F1 | Parameters | Strategy |
|:------|:------------:|:--------:|:---------:|:---------|
| **EfficientNetV2-S** | **1.0000** | **1.0000** | 20.2M | 2-phase fine-tuning |
| **ConvNeXt-Tiny** | **1.0000** | **1.0000** | 27.8M | Gradual unfreezing |
| MobileNetV3-Large | 0.9989 | 0.9990 | 4.2M | Lightweight & fast |
| DenseNet-121 | 0.9989 | 0.9990 | 7.0M | Feature reuse |

**Training details:**

- **Phase 1** (5 epochs): Backbone frozen — only classifier head trained (LR: 1e-3)
- **Phase 2** (15 epochs): Gradual unfreezing — discriminative learning rates (LR: 1e-4 → 1e-5)
- **Augmentation:** Albumentations (MixUp, CutMix, RandAugment, CoarseDropout, GridDistortion, etc.)
- **Optimizer:** AdamW (weight decay: 0.01) + CosineAnnealingWarmRestarts
- **Label Smoothing:** α = 0.1 · **Mixed Precision (AMP)** · Early Stopping (patience=7)

<details>
<summary>📸 Training charts</summary>

| | |
|:---:|:---:|
| ![Training History](03_cnn_transfer_learning/results/training_history_all.png) | ![CNN Comparison](03_cnn_transfer_learning/results/cnn_comparison_chart.png) |

</details>

---

### 4 · Vision Transformers & Hybrid Models

> `04_vision_transformers/04_vision_transformers.ipynb`

#### Pure Transformers

| Model | Patch | Test Accuracy | Parameters |
|:------|:-----:|:------------:|:---------:|
| **ViT-Small/16** | 16×16 | **1.0000** | 21.7M |
| DeiT-Small | 16×16 | ~0.95–0.99 | — |
| Swin-Tiny | 4×4 window | 0.9946 | 27.5M |

#### Hybrid Models

| Model | Type | Test Accuracy | Parameters |
|:------|:-----|:------------:|:---------:|
| **CoAtNet-0** | CNN + Attention | **1.0000** | 26.7M |
| **EfficientFormer-L1** | Hybrid | **1.0000** | **11.4M** |
| MaxViT-Tiny | Multi-axis attention | ~0.99–1.00 | — |

> 💡 **EfficientFormer-L1** was the **most efficient model**, achieving 100% accuracy with only 11.4M parameters.

**Advanced analysis:** Attention map visualization · SHAP DeepExplainer · Knowledge Distillation (teacher → 388 KB student model)

<details>
<summary>📸 Attention maps</summary>

| | |
|:---:|:---:|
| ![ViT Comparison](04_vision_transformers/results/vit_comparison_chart.png) | ![Attention Maps](04_vision_transformers/results/vit_attention_samples.png) |

</details>

---

### 5 · Advanced Techniques

> `05_advanced_techniques/05_advanced_techniques.ipynb`

| Technique | Category | Test Accuracy | Description |
|:----------|:---------|:------------:|:------------|
| Cross-Entropy (Baseline) | Supervised | 0.9258 | Standard training |
| **SimCLR** | Self-Supervised | **0.9076** | No labels — NT-Xent loss |
| Triplet Loss | Metric Learning | ~0.90–0.92 | Embedding space (margin=0.5) |
| ArcFace | Metric Learning | ~0.90–0.92 | Angular margin (s=30, m=0.5) |
| Focal Loss | Loss Engineering | 0.8785 | Hard example focus with γ=2.0 |
| NAS (Optuna) | Architecture Search | 0.8704 | Best: 4 layers, 64 filters |
| FPN Multi-Scale | Feature Fusion | 0.6910 | Multi-scale pyramid |

**SimCLR details:** ResNet-18 backbone · Projection head (512 → 128) · τ = 0.5 · Dual-view augmentation · **90.76% accuracy without any labels**

<details>
<summary>📸 Advanced technique results</summary>

| | |
|:---:|:---:|
| ![SimCLR t-SNE](05_advanced_techniques/results/simclr_tsne.png) | ![Metric Learning](05_advanced_techniques/results/metric_learning_losses.png) |
| ![Focal vs CE](05_advanced_techniques/results/focal_vs_ce_loss.png) | ![Advanced Methods](05_advanced_techniques/results/advanced_methods_chart.png) |

</details>

---

### 6 · Ensemble Model & Final Analysis

> `06_ensemble/06_ensemble.ipynb`

| Strategy | Method | Test Accuracy | Cohen's κ | Macro F1 |
|:---------|:-------|:------------:|:---------:|:--------:|
| **Hard Voting** | Majority vote | **1.0000** | **1.0000** | **1.0000** |
| **Soft Voting** | Probability averaging | **1.0000** | **1.0000** | **1.0000** |
| **Weighted Soft** | Accuracy² weighting | **1.0000** | **1.0000** | **1.0000** |
| **Stacking (XGBoost)** | Meta-learner | **1.0000** | **1.0000** | **1.0000** |
| Stacking (LightGBM) | Meta-learner | 0.9968 | 0.9965 | 0.9969 |
| Stacking (MLP) | Neural meta-learner | 0.9968 | 0.9965 | ~0.997 |
| Rank Averaging | Rank-based | 0.9968 | 0.9965 | 0.9969 |
| **Greedy Selection** | Iterative selection | **1.0000** | **1.0000** | **1.0000** |

**Statistical tests:** McNemar test (pairwise comparison) · Friedman test (multi-classifier comparison)  
**Deployment:** ONNX export · INT8 Quantization

<details>
<summary>📸 Ensemble results</summary>

| | |
|:---:|:---:|
| ![Confusion Matrix](06_ensemble/results/confusion_matrix_comparison-2.png) | ![Per-class F1](06_ensemble/results/per_class_f1-2.png) |
| ![ROC-AUC](06_ensemble/results/roc_auc_curves-2.png) | ![Error Correlation](06_ensemble/results/error_correlation_matrix-2.png) |

</details>

---

## 📈 Performance Comparison

```
                             Accuracy (%)
  Classical ML (RF)      ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░  63.33
  Custom CNN (DW-Sep)    █████████████████████████████░░░░░░░░░░░░░░  92.98
  SimCLR (no labels)     ████████████████████████████░░░░░░░░░░░░░░░  90.76
  Transfer Learning      ████████████████████████████████████████████ 100.00
  Vision Transformers    ████████████████████████████████████████████ 100.00
  Ensemble               ████████████████████████████████████████████ 100.00
```

### Complete Model Leaderboard

| # | Model | Type | Accuracy | F1 | Parameters |
|:-:|:------|:-----|:--------:|:--:|:---------:|
| 1 | SVM-Polynomial | Classical | 0.1667 | 0.016 | — |
| 2 | KNN (k=5) | Classical | 0.3333 | 0.168 | — |
| 3 | CatBoost | Classical | 0.3333 | 0.156 | — |
| 4 | XGBoost | Classical | 0.3667 | 0.198 | — |
| 5 | SVM-RBF | Classical | 0.4000 | 0.231 | — |
| 6 | Logistic Regression | Classical | 0.4333 | 0.281 | — |
| 7 | Gaussian NB | Classical | 0.4333 | 0.260 | — |
| 8 | LightGBM | Classical | 0.4667 | 0.296 | — |
| 9 | SVM-Linear | Classical | 0.5000 | 0.310 | — |
| 10 | **Random Forest** | Classical | **0.6333** | **0.452** | 200 trees |
| 11 | FPN Multi-Scale | Advanced | 0.6910 | — | — |
| 12 | NAS (Optuna) | Advanced | 0.8704 | — | 4 layers |
| 13 | Focal Loss | Advanced | 0.8785 | — | γ=2.0 |
| 14 | ResidualCNN | CNN | 0.8801 | 0.838 | 2.1M |
| 15 | SimpleCNN | CNN | 0.8855 | 0.855 | 2.5M |
| 16 | SimCLR | Self-Sup | 0.9076 | — | ResNet-18 |
| 17 | Cross-Entropy Baseline | Advanced | 0.9258 | — | — |
| 18 | DepthwiseSep CNN | CNN | 0.9298 | 0.917 | **0.2M** |
| 19 | Swin-Tiny | ViT | 0.9946 | ~0.994 | 27.5M |
| 20 | MobileNetV3-Large | Transfer | 0.9989 | 0.999 | 4.2M |
| 21 | DenseNet-121 | Transfer | 0.9989 | 0.999 | 7.0M |
| 22 | EfficientNetV2-S | Transfer | **1.0000** | **1.000** | 20.2M |
| 23 | ConvNeXt-Tiny | Transfer | **1.0000** | **1.000** | 27.8M |
| 24 | ViT-Small/16 | ViT | **1.0000** | **1.000** | 21.7M |
| 25 | CoAtNet-0 | Hybrid | **1.0000** | **1.000** | 26.7M |
| 26 | EfficientFormer-L1 | Hybrid | **1.0000** | **1.000** | 11.4M |
| 27 | Ensemble (Voting) | Ensemble | **1.0000** | **1.000** | — |
| 28 | Ensemble (Stacking) | Ensemble | **1.0000** | **1.000** | — |

---

## 🔬 Key Findings & Contributions

1. **Classical vs. Deep Learning Gap (~37 points):** Classical ML with handcrafted features peaked at 63.33% accuracy, while transfer learning reached 100%.

2. **Power of Transfer Learning:** ImageNet pre-trained models outperformed custom CNNs by **~7–12 percentage points**.

3. **Transformers Match CNNs:** ViT and hybrid models achieved the same 100% accuracy as CNN-based transfer learning.

4. **Self-Supervised Learning Potential:** SimCLR achieved 90.76% accuracy without using any labels, demonstrating its potential to reduce annotation costs.

5. **Efficiency–Performance Trade-off:** EfficientFormer-L1 (11.4M params) and MobileNetV3 (4.2M params) are ideal candidates for embedded deployment.

6. **Ensemble Reliability:** Multiple ensemble strategies achieved perfect classification with Cohen's κ = 1.0.

---

## 🧰 Tech Stack

| Category | Tools |
|:---------|:------|
| **Deep Learning** | PyTorch · torchvision · timm |
| **Classical ML** | scikit-learn · XGBoost · LightGBM · CatBoost |
| **Image Processing** | OpenCV · Albumentations · scikit-image |
| **Visualization** | Matplotlib · Seaborn · Plotly |
| **Explainability** | Grad-CAM · SHAP · LIME · Captum |
| **Optimization** | Optuna |
| **Dimensionality Reduction** | t-SNE · UMAP · PCA · LDA |
| **Deployment** | ONNX · ONNX Runtime |
| **Experiment Tracking** | Weights & Biases (wandb) |

---

## ⚙️ Installation

### Requirements

- Python ≥ 3.10
- CUDA-capable GPU (recommended; also runs on CPU)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/emirsecer1/vegetable-classification-thesis.git
cd vegetable-classification-thesis

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Kaggle Environment

1. Add the [Vegetables dataset](https://www.kaggle.com/datasets/emirsecer/vegetables) to your Kaggle notebook.
2. The `DATA_DIR` variable is automatically configured for the Kaggle environment:
   ```python
   DATA_DIR = "../input/vegetables/SEBZE/"
   ```
3. Run the notebooks in order: `01` → `02` → `03` → `04` → `05` → `06`

### Local Environment

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/emirsecer/vegetables) and extract it.
2. Update the `DATA_DIR` variable at the beginning of each notebook:
   ```python
   DATA_DIR = "/path/to/your/SEBZE/"
   ```
3. If the data is not found, notebooks run in **demo mode** (with synthetic data).

### Thesis Report

The compiled thesis report is available in the `report/` directory:

| Format | File |
|:-------|:-----|
| PDF | [`report/main.pdf`](report/main.pdf) |
| Word | [`report/main.docx`](report/main.docx) |
| LaTeX source | [`report/main.tex`](report/main.tex) |

---

## 📚 References

<details>
<summary>View all academic references (30 sources)</summary>

1. He, K. et al. "Deep Residual Learning for Image Recognition." *CVPR*, 2016.
2. Dosovitskiy, A. et al. "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale." *ICLR*, 2021.
3. Liu, Z. et al. "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows." *ICCV*, 2021.
4. Tan, M. & Le, Q. "EfficientNetV2: Smaller Models and Faster Training." *ICML*, 2021.
5. Howard, A. et al. "Searching for MobileNetV3." *ICCV*, 2019.
6. Liu, Z. et al. "A ConvNet for the 2020s." *CVPR*, 2022.
7. Huang, G. et al. "Densely Connected Convolutional Networks." *CVPR*, 2017.
8. Chen, T. et al. "A Simple Framework for Contrastive Learning of Visual Representations." *ICML*, 2020.
9. Deng, J. et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." *CVPR*, 2019.
10. Lin, T.-Y. et al. "Focal Loss for Dense Object Detection." *ICCV*, 2017.
11. Lin, T.-Y. et al. "Feature Pyramid Networks for Object Detection." *CVPR*, 2017.
12. Breiman, L. "Random Forests." *Machine Learning*, 45(1), 5–32, 2001.
13. Cortes, C. & Vapnik, V. "Support-Vector Networks." *Machine Learning*, 20(3), 273–297, 1995.
14. Chen, T. & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." *KDD*, 2016.
15. Ke, G. et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS*, 2017.
16. Dalal, N. & Triggs, B. "Histograms of Oriented Gradients for Human Detection." *CVPR*, 2005.
17. Ojala, T. et al. "Multiresolution Gray-Scale and Rotation Invariant Texture Classification with LBP." *IEEE TPAMI*, 2002.
18. Haralick, R. M. et al. "Textural Features for Image Classification." *IEEE TSMC*, 1973.
19. Van der Maaten, L. & Hinton, G. "Visualizing Data Using t-SNE." *JMLR*, 9, 2579–2605, 2008.
20. McInnes, L. et al. "UMAP: Uniform Manifold Approximation and Projection." *arXiv:1802.03426*, 2018.
21. Akiba, T. et al. "Optuna: A Next-Generation Hyperparameter Optimization Framework." *KDD*, 2019.
22. Hinton, G. et al. "Distilling the Knowledge in a Neural Network." *arXiv:1503.02531*, 2015.
23. Lundberg, S. M. & Lee, S.-I. "A Unified Approach to Interpreting Model Predictions." *NeurIPS*, 2017.
24. Dai, Z. et al. "CoAtNet: Marrying Convolution and Attention for All Data Sizes." *NeurIPS*, 2021.
25. Li, Y. et al. "EfficientFormer: Vision Transformers at MobileNet Speed." *NeurIPS*, 2022.
26. Schroff, F. et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering." *CVPR*, 2015.
27. Bengio, Y. et al. "Curriculum Learning." *ICML*, 2009.
28. Wolpert, D. H. "Stacked Generalization." *Neural Networks*, 5(2), 241–259, 1992.
29. Loshchilov, I. & Hutter, F. "Decoupled Weight Decay Regularization." *ICLR*, 2019.
30. Touvron, H. et al. "Training Data-Efficient Image Transformers & Distillation Through Attention." *ICML*, 2021.

</details>

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Emir Seçer — Undergraduate Thesis, 2025</sub>
</p>
