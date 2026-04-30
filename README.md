# Tomato Leaf Disease Classifier

> A deep learning pipeline for identifying tomato leaf diseases from images, built on **EfficientNetB0** and trained on the PlantVillage dataset across 10 classes — 9 disease categories and 1 healthy baseline.

---

## Overview

Crop disease detection at scale is a hard problem. This project provides a clean, modular image classification pipeline that can identify 10 tomato leaf conditions from a single photo, with a model architecture tuned for high accuracy and lightweight enough to deploy on mobile or edge devices via TensorFlow Lite.

---

## Classes

The model is trained exclusively on tomato leaf data filtered from the PlantVillage dataset:

| # | Class |
|---|-------|
| 1 | Bacterial Spot |
| 2 | Early Blight |
| 3 | Late Blight |
| 4 | Leaf Mold |
| 5 | Septoria Leaf Spot |
| 6 | Spider Mites (Two-spotted) |
| 7 | Target Spot |
| 8 | Tomato Mosaic Virus |
| 9 | Yellow Leaf Curl Virus |
| 10 | Healthy |

---

## Model Architecture

Built on **EfficientNetB0** pretrained on ImageNet, with a custom classification head:

```
EfficientNetB0 (frozen base — feature extraction)
    └── GlobalAveragePooling2D
    └── Dense(256, relu)
    └── Dropout(0.3)
    └── Dense(10, softmax)
```

**Training configuration:**
- Optimizer: Adam
- Loss: Categorical cross-entropy
- Augmentation: Random rotation, zoom, horizontal/vertical flips
- Baseline training: 10+ epochs with fine-tuning support

---

## Repository Structure

```
Tomato-Disease-Classifier/
├── Tomato_Dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── src/
│   ├── dataset_split.py       # Splits raw dataset into train/val/test
│   ├── train_model.py         # Model definition, training, and checkpointing
│   ├── evaluate.py            # Metrics: accuracy, confusion matrix, loss curves
│   └── predict.py             # Single-image inference via CLI
├── models/
│   └── tomato_disease_classifier.h5
├── requirements.txt
└── README.md
```

---

## Quick Start

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Prepare the dataset**

Organizes raw PlantVillage tomato images into `train/`, `val/`, and `test/` splits:

```bash
python src/dataset_split.py
```

**3. Train the model**

```bash
python src/train_model.py
```

**4. Evaluate performance**

Outputs test accuracy, confusion matrix, and training curves:

```bash
python src/evaluate.py
```

**5. Run inference on a single image**

```bash
python src/predict.py --image path/to/leaf.jpg
```

---

## Results

| Metric | Value |
|--------|-------|
| Training Accuracy | — |
| Validation Accuracy | — |
| Test Accuracy | — |

> Update this table after training. Evaluation artifacts (confusion matrix, loss/accuracy curves) are saved automatically by `evaluate.py`.

---

## Deployment

The trained model is ready for deployment across several targets:

- **TensorFlow Lite** — Convert `.h5` to `.tflite` for Android/iOS apps
- **Flask / FastAPI** — Wrap the model in a REST endpoint for web integration
- **ONNX** — Export for cross-platform and non-TensorFlow inference runtimes
- **Mobile agriculture assistant** — Designed to run efficiently on-device

---

## Requirements

```
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
```

---

## License

Distributed under the [MIT License](LICENSE).

---

## Acknowledgments

- **PlantVillage** — For the open-access, high-quality agricultural image dataset
- **Google AI / EfficientNet authors** — For the EfficientNet architecture and pretrained weights
- **TensorFlow team** — For the model development and deployment ecosystem
