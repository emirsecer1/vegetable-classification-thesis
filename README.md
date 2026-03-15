# 🥬 Sebze Sınıflandırma - Bitirme Tezi

Bu proje, Kaggle'daki **"Vegetables"** veri seti üzerinde kapsamlı bir görüntü sınıflandırma çalışmasıdır. Bitirme tezi kapsamında 6 ayrı notebook'ta farklı yaklaşımlar uygulanmış ve son olarak bir ensemble modeli kurulmuştur.

## 📊 Dataset Bilgisi

| Özellik | Değer |
|---------|-------|
| Veri Seti | Kaggle Vegetables |
| Ana Klasör | SEBZE |
| Sınıf Sayısı | 23 |
| Toplam Görsel | 6170 |
| Problem Tipi | Çok Sınıflı Görüntü Sınıflandırma |

## 🗂️ Klasör Yapısı

```
vegetable-classification-thesis/
├── README.md
├── requirements.txt
├── 01_eda_feature_engineering/
│   ├── results/
│   └── 01_eda_feature_engineering.ipynb
├── 02_classical_ml/
│   ├── results/
│   └── 02_classical_ml.ipynb
├── 03_cnn_transfer_learning/
│   ├── results/
│   └── 03_cnn_transfer_learning.ipynb
├── 04_vision_transformers/
│   ├── results/
│   └── 04_vision_transformers.ipynb
├── 05_advanced_techniques/
│   ├── results/
│   └── 05_advanced_techniques.ipynb
└── 06_ensemble/
    ├── results/
    └── 06_ensemble.ipynb
```

## 📓 Notebook Açıklamaları

### 1. EDA ve Özellik Çıkarımı (`01_eda_feature_engineering`)
Keşifsel veri analizi ve el yapımı özellik çıkarımı:
- Sınıf dağılımı, boyut ve renk analizleri
- Color Histogram (HSV, LAB), LBP, GLCM, HOG, Hu Moments
- Edge özellikleri (Canny, Sobel), İstatistiksel özellikler
- FFT tabanlı frekans domain özellikleri
- t-SNE ve UMAP görselleştirme

### 2. Klasik Makine Öğrenmesi (`02_classical_ml`)
El yapımı özelliklerle klasik ML modelleri:
- SVM (RBF, Polynomial, Linear kernel)
- Random Forest, XGBoost, LightGBM, CatBoost
- KNN, Logistic Regression, Naive Bayes
- PCA, LDA boyut indirgeme
- Optuna ile hyperparameter tuning

### 3. CNN ve Transfer Learning (`03_cnn_transfer_learning`)
Derin öğrenme tabanlı sınıflandırma:
- Özel CNN mimarileri (SimpleCNN, ResidualCNN, DepthwiseSeparableCNN)
- Transfer Learning: EfficientNetV2-S, ConvNeXt-Tiny, RegNetY, MobileNetV3, DenseNet-121
- Albumentations ile veri artırma (MixUp, CutMix, RandAugment)
- Grad-CAM görselleştirme

### 4. Vision Transformers (`04_vision_transformers`)
Transformer tabanlı modeller:
- ViT-Small/16, DeiT-Small, Swin-Tiny, BEiT-Base
- Hibrit modeller: CoAtNet-0, MaxViT-Tiny, EfficientFormer-L1
- Attention map görselleştirme
- SHAP açıklanabilirlik analizi
- Knowledge Distillation

### 5. Gelişmiş Teknikler (`05_advanced_techniques`)
Modern öğrenme yaklaşımları:
- SimCLR (Contrastive Self-Supervised Learning)
- Metric Learning: Triplet Loss, ArcFace
- Curriculum Learning, Focal Loss
- Multi-Scale Feature Fusion (FPN)
- Neural Architecture Search (NAS) ile Optuna

### 6. Ensemble (`06_ensemble`)
Tüm modellerin birleştirilmesi:
- Hard/Soft Voting, Weighted Averaging
- Stacking (XGBoost, LightGBM, MLP meta-learner)
- Blending, Rank Averaging
- McNemar testi, Friedman testi
- ONNX export ve INT8 Quantization

## 🛠️ Kurulum

```bash
pip install -r requirements.txt
```

## 🚀 Kullanım Talimatları

### Kaggle'da Kullanım
1. Kaggle'da [Vegetables](https://www.kaggle.com/datasets) veri setini bulun ve notebook'a ekleyin.
2. Notebook'lardaki `DATA_DIR` değişkeni Kaggle ortamı için otomatik ayarlanmaktadır:
   ```python
   DATA_DIR = "../input/vegetables/SEBZE/"
   ```
3. Notebook'ları sırasıyla çalıştırın (01 → 02 → ... → 06)

### Yerel Ortamda Kullanım
1. Kaggle'dan veri setini indirin ve bir klasöre çıkarın.
2. Her notebook'un başındaki `DATA_DIR` değişkenini kendi yolunuzla güncelleyin:
   ```python
   DATA_DIR = "/path/to/your/SEBZE/"
   ```
3. Notebook'lar veri dizini bulunamazsa demo modunda çalışır (sentetik veri ile).

## 🧰 Kullanılan Teknolojiler

| Kategori | Kütüphaneler |
|----------|-------------|
| Derin Öğrenme | PyTorch, timm, torchvision |
| Görüntü İşleme | OpenCV, Albumentations, scikit-image |
| Klasik ML | scikit-learn, XGBoost, LightGBM, CatBoost |
| Görselleştirme | Matplotlib, Seaborn, Plotly |
| Açıklanabilirlik | pytorch-grad-cam, SHAP, LIME, captum |
| Optimizasyon | Optuna |
| Dışa Aktarım | ONNX, ONNX Runtime |

## 📈 Sonuçlar

| Model | Accuracy | Macro-F1 | Params |
|-------|----------|----------|--------|
| SVM (RBF) | - | - | - |
| Random Forest | - | - | - |
| XGBoost | - | - | - |
| SimpleCNN | - | - | - |
| EfficientNetV2-S | - | - | - |
| Swin-Tiny | - | - | - |
| Ensemble (Best) | - | - | - |

*Sonuçlar notebook'lar çalıştırıldıkça doldurulacaktır.*

## 📚 Referanslar

- Dosovitskiy, A., et al. "An image is worth 16x16 words: Transformers for image recognition at scale." ICLR 2021.
- Liu, Z., et al. "Swin transformer: Hierarchical vision transformer using shifted windows." ICCV 2021.
- Tan, M., & Le, Q. "EfficientNetV2: Smaller models and faster training." ICML 2021.
- Chen, T., et al. "A simple framework for contrastive learning of visual representations." ICML 2020.
- Guo, C., et al. "On calibration of modern neural networks." ICML 2017.

## 📄 Lisans

MIT License - Detaylar için `LICENSE` dosyasına bakınız.
