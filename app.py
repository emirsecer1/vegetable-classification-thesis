"""
Sebze Sınıflandırma Tahmin Arayüzü
===================================
Eğitilmiş modelleri kullanarak sebze görüntülerini sınıflandırır.
Tüm .pth model dosyalarını otomatik olarak yükler ve karşılaştırır.
"""

import os
import time
import glob

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ──────────────────────────────────────────────
# Sabitler
# ──────────────────────────────────────────────
NUM_CLASSES = 23
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sınıf isimleri (veri setindeki klasör adlarına göre alfabetik sıralı)
CLASS_NAMES = [
    "Bean",
    "Bitter Gourd",
    "Bottle Gourd",
    "Brinjal",
    "Broccoli",
    "Cabbage",
    "Capsicum",
    "Carrot",
    "Cauliflower",
    "Cucumber",
    "Papaya",
    "Potato",
    "Pumpkin",
    "Radish",
    "Tomato",
]
# Veri setinde 23 sınıf var; bilinmeyen sınıflar için genel etiketler
while len(CLASS_NAMES) < NUM_CLASSES:
    CLASS_NAMES.append(f"Sınıf {len(CLASS_NAMES) + 1}")

# ──────────────────────────────────────────────
# Görüntü ön işleme
# ──────────────────────────────────────────────
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ──────────────────────────────────────────────
# Özel CNN model tanımları
# ──────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResidualCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResidualBlock(64), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 1), ResidualBlock(128), nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 1), ResidualBlock(256), nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.classifier(x)


class DepthwiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class DepthwiseSeparableCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            DepthwiseConv(32, 64, stride=2),
            DepthwiseConv(64, 128, stride=2),
            DepthwiseConv(128, 256, stride=2),
            DepthwiseConv(256, 512, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class StudentCNN(nn.Module):
    """Knowledge Distillation öğrenci modeli."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# ──────────────────────────────────────────────
# Model kayıt defteri
# ──────────────────────────────────────────────
# Her model için: (görünen ad, mimari oluşturma fonksiyonu, .pth yolu)
MODEL_REGISTRY: dict[str, dict] = {}


def _register(display_name: str, pth_relative: str, build_fn):
    """Bir modeli kayıt defterine ekler (dosya varsa)."""
    pth_path = os.path.join(BASE_DIR, pth_relative)
    if os.path.isfile(pth_path):
        MODEL_REGISTRY[display_name] = {
            "path": pth_path,
            "build_fn": build_fn,
            "size_mb": os.path.getsize(pth_path) / (1024 * 1024),
        }


# --- Özel CNN modelleri (Notebook 03) ---
_register(
    "SimpleCNN",
    "03_cnn_transfer_learning/results/SimpleCNN_best.pth",
    lambda: SimpleCNN(NUM_CLASSES),
)
_register(
    "ResidualCNN",
    "03_cnn_transfer_learning/results/ResidualCNN_best.pth",
    lambda: ResidualCNN(NUM_CLASSES),
)
_register(
    "DepthwiseSeparableCNN",
    "03_cnn_transfer_learning/results/DepthwiseSeparableCNN_best.pth",
    lambda: DepthwiseSeparableCNN(NUM_CLASSES),
)

# --- Transfer Learning modelleri (Notebook 03) ---
_register(
    "MobileNetV3-Large",
    "03_cnn_transfer_learning/results/MobileNetV3_Large_best.pth",
    lambda: timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=NUM_CLASSES),
)

# --- Vision Transformer modelleri (Notebook 04) ---
_register(
    "ViT-Small/16",
    "04_vision_transformers/results/ViT_Small_16_best.pth",
    lambda: timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=NUM_CLASSES),
)
_register(
    "Swin-Tiny",
    "04_vision_transformers/results/Swin_Tiny_best.pth",
    lambda: timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=NUM_CLASSES),
)
_register(
    "CoAtNet-0",
    "04_vision_transformers/results/CoAtNet_0_best.pth",
    lambda: timm.create_model("coatnet_0_rw_224", pretrained=False, num_classes=NUM_CLASSES),
)
_register(
    "EfficientFormer-L1",
    "04_vision_transformers/results/EfficientFormer_L1_best.pth",
    lambda: timm.create_model("efficientformer_l1", pretrained=False, num_classes=NUM_CLASSES),
)
_register(
    "KD-Student",
    "04_vision_transformers/results/KD_student_best.pth",
    lambda: StudentCNN(NUM_CLASSES),
)


# ──────────────────────────────────────────────
# Model yükleme ve önbellekleme
# ──────────────────────────────────────────────
_loaded_models: dict[str, nn.Module] = {}


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_model(name: str) -> nn.Module:
    """Modeli yükler ve önbelleğe alır."""
    if name in _loaded_models:
        return _loaded_models[name]

    info = MODEL_REGISTRY[name]
    model = info["build_fn"]()
    state_dict = torch.load(info["path"], map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()

    info["params"] = _count_parameters(model)
    _loaded_models[name] = model
    return model


# ──────────────────────────────────────────────
# Tahmin fonksiyonları
# ──────────────────────────────────────────────
def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Görüntüyü model girdisine dönüştürür."""
    if image is None:
        raise ValueError("Görüntü yüklenemedi.")

    # Gradio RGB olarak gönderir
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    augmented = val_transform(image=image)
    tensor = augmented["image"].unsqueeze(0).to(DEVICE)
    return tensor


def predict_single(model_name: str, image: np.ndarray):
    """Tek bir model ile tahmin yapar."""
    if image is None:
        return "⚠️ Lütfen bir görüntü yükleyin.", None

    model = load_model(model_name)
    tensor = preprocess_image(image)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    elapsed = time.perf_counter() - start

    probs = F.softmax(logits, dim=1)[0]
    top5_probs, top5_idx = probs.topk(5)

    info = MODEL_REGISTRY[model_name]

    # Gradio label çıktısı
    label_dict = {}
    for prob, idx in zip(top5_probs, top5_idx):
        class_name = CLASS_NAMES[idx.item()]
        label_dict[class_name] = float(prob)

    details = (
        f"📊 **Model:** {model_name}\n"
        f"⏱️ **Çıkarım Süresi:** {elapsed * 1000:.1f} ms\n"
        f"📦 **Model Boyutu:** {info['size_mb']:.1f} MB\n"
        f"🔢 **Parametre Sayısı:** {info.get('params', 0):,}\n"
    )

    return label_dict, details


def predict_all_models(image: np.ndarray):
    """Tüm modeller ile tahmin yapar ve karşılaştırır."""
    if image is None:
        return "⚠️ Lütfen bir görüntü yükleyin."

    tensor = preprocess_image(image)
    results = []

    for name in MODEL_REGISTRY:
        model = load_model(name)
        info = MODEL_REGISTRY[name]

        start = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
        elapsed = time.perf_counter() - start

        probs = F.softmax(logits, dim=1)[0]
        top_prob, top_idx = probs.topk(1)
        predicted_class = CLASS_NAMES[top_idx[0].item()]
        confidence = float(top_prob[0]) * 100

        top3_probs, top3_idx = probs.topk(3)
        top3_str = ", ".join(
            f"{CLASS_NAMES[i.item()]} (%{p.item() * 100:.1f})"
            for p, i in zip(top3_probs, top3_idx)
        )

        results.append({
            "name": name,
            "predicted": predicted_class,
            "confidence": confidence,
            "time_ms": elapsed * 1000,
            "size_mb": info["size_mb"],
            "params": info.get("params", 0),
            "top3": top3_str,
        })

    # Sonuçları çıkarım süresine göre sırala
    results.sort(key=lambda r: r["time_ms"])

    # Markdown tablo oluştur
    lines = [
        "## 🏆 Model Karşılaştırma Sonuçları\n",
        "| # | Model | Tahmin | Güven (%) | Süre (ms) | Boyut (MB) | Parametre |",
        "|---|-------|--------|-----------|-----------|------------|-----------|",
    ]
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | **{r['name']}** | {r['predicted']} | "
            f"{r['confidence']:.1f} | {r['time_ms']:.1f} | "
            f"{r['size_mb']:.1f} | {r['params']:,} |"
        )

    lines.append("\n---\n### 📋 Detaylı Top-3 Tahminler\n")
    for r in results:
        lines.append(f"- **{r['name']}**: {r['top3']}")

    # Tahmin uyumu analizi
    predictions = [r["predicted"] for r in results]
    unique_preds = set(predictions)
    lines.append("\n---\n### 🔍 Tahmin Uyumu Analizi\n")
    if len(unique_preds) == 1:
        lines.append(
            f"✅ **Tüm modeller aynı sınıfı tahmin etti: {predictions[0]}**"
        )
    else:
        lines.append(f"⚠️ **Modeller farklı tahminlerde bulundu!**\n")
        for pred in unique_preds:
            agreeing = [r["name"] for r in results if r["predicted"] == pred]
            lines.append(f"- **{pred}**: {', '.join(agreeing)}")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Gradio Arayüzü
# ──────────────────────────────────────────────
model_names = list(MODEL_REGISTRY.keys())

if not model_names:
    raise RuntimeError(
        "Hiçbir .pth model dosyası bulunamadı! "
        "Lütfen results/ klasörlerindeki model dosyalarını kontrol edin."
    )

with gr.Blocks(title="🥬 Sebze Sınıflandırma") as demo:
    gr.Markdown(
        "# 🥬 Sebze Görüntüsü Sınıflandırma Arayüzü\n"
        "Eğitilmiş derin öğrenme modellerini kullanarak sebze görüntülerini sınıflandırın.\n"
        f"**{len(model_names)} model yüklendi:** {', '.join(model_names)}"
    )

    with gr.Tabs():
        # ── Sekme 1: Tek Model Tahmini ──
        with gr.Tab("🎯 Tek Model Tahmini"):
            with gr.Row():
                with gr.Column(scale=1):
                    single_image = gr.Image(
                        label="Sebze Görüntüsü Yükleyin",
                        type="numpy",
                    )
                    single_model = gr.Dropdown(
                        choices=model_names,
                        value=model_names[0],
                        label="Model Seçin",
                    )
                    single_btn = gr.Button("🔍 Tahmin Et", variant="primary")

                with gr.Column(scale=1):
                    single_label = gr.Label(
                        label="Tahmin Sonuçları (Top-5)", num_top_classes=5
                    )
                    single_details = gr.Markdown(label="Model Bilgileri")

            single_btn.click(
                fn=predict_single,
                inputs=[single_model, single_image],
                outputs=[single_label, single_details],
            )

        # ── Sekme 2: Tüm Modelleri Karşılaştır ──
        with gr.Tab("⚡ Tüm Modelleri Karşılaştır"):
            with gr.Row():
                with gr.Column(scale=1):
                    multi_image = gr.Image(
                        label="Sebze Görüntüsü Yükleyin",
                        type="numpy",
                    )
                    multi_btn = gr.Button(
                        "🚀 Tüm Modellerle Tahmin Et", variant="primary"
                    )

                with gr.Column(scale=2):
                    multi_output = gr.Markdown(label="Karşılaştırma Sonuçları")

            multi_btn.click(
                fn=predict_all_models,
                inputs=[multi_image],
                outputs=[multi_output],
            )

        # ── Sekme 3: Model Bilgileri ──
        with gr.Tab("📊 Model Bilgileri"):
            info_lines = [
                "## Mevcut Modeller\n",
                "| Model | Dosya Boyutu (MB) | Kaynak |",
                "|-------|-------------------|--------|",
            ]
            sources = {
                "SimpleCNN": "Özel CNN (Notebook 03)",
                "ResidualCNN": "Özel CNN (Notebook 03)",
                "DepthwiseSeparableCNN": "Özel CNN (Notebook 03)",
                "MobileNetV3-Large": "Transfer Learning (Notebook 03)",
                "ViT-Small/16": "Vision Transformer (Notebook 04)",
                "Swin-Tiny": "Vision Transformer (Notebook 04)",
                "CoAtNet-0": "Hibrit Model (Notebook 04)",
                "EfficientFormer-L1": "Hibrit Model (Notebook 04)",
                "KD-Student": "Knowledge Distillation (Notebook 04)",
            }
            for name, info in MODEL_REGISTRY.items():
                source = sources.get(name, "—")
                info_lines.append(
                    f"| {name} | {info['size_mb']:.1f} | {source} |"
                )

            info_lines.append(
                "\n### Sınıf Listesi (23 Sebze)\n"
                + ", ".join(f"`{c}`" for c in CLASS_NAMES)
            )

            gr.Markdown("\n".join(info_lines))

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
