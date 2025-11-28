# 🍅 Tomato Leaf Disease Classification

### Deep Learning Project using PlantVillage Tomato Dataset

This project aims to build a **deep learning model** that can classify tomato leaf diseases from images using the **PlantVillage dataset**. The dataset contains 10 tomato classes — 9 disease categories and 1 healthy category.
The model is built using **TensorFlow** and **EfficientNetB0**, achieving high accuracy on plant disease recognition.

---

## 📌 Project Objectives

* Train a robust **image classification model** for tomato leaf diseases
* Use **EfficientNetB0** for feature extraction and fine-tuning
* Provide a clean, modular pipeline for:

  * Dataset preparation
  * Training
  * Validation
  * Testing
  * Single image prediction
* Deployable model (Flask/TF-Lite ready)

---

## 📂 Dataset Information

You filtered the dataset to include only **tomato leaf classes** from PlantVillage:

```
Tomato_Late_blight  
Tomato_Leaf_Mold  
Tomato_Septoria_leaf_spot  
Tomato_Early_blight  
Tomato_Spider_mites_Two_spotted_spider_mite  
Tomato__Target_Spot  
Tomato_Bacterial_spot  
Tomato__Tomato_mosaic_virus  
Tomato__Tomato_YellowLeaf__Curl_Virus  
Tomato_healthy  
```

Total Classes → **10**

---

## 🧱 Project Structure

```
Tomato-Disease-Classifier/
│
├── Tomato_Dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── src/
│   ├── dataset_split.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── predict.py
│
├── models/
│   └── tomato_disease_classifier.h5
│
├── README.md
└── requirements.txt
```

---

## 🔧 Setup Instructions

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Dataset Split (train/val/test)

Run this script to organize dataset:

```bash
python src/dataset_split.py
```

### 3️⃣ Train Model

```bash
python src/train_model.py
```

### 4️⃣ Evaluate Model

```bash
python src/evaluate.py
```

### 5️⃣ Predict a Single Image

```bash
python src/predict.py --image leaf.jpg
```

---

## 🧠 Model Architecture

The model uses:

* **EfficientNetB0** (pretrained on ImageNet)
* Frozen base layers for feature extraction
* Custom classification head:

  * GlobalAveragePooling
  * Dense (256)
  * Dropout (0.3)
  * Dense (10, softmax)

Training includes:

* Data Augmentation (rotation, zoom, flips)
* Adam optimizer
* Categorical cross-entropy loss
* 10+ epochs baseline training

---

## 📊 Results

Metrics evaluated:

* Training accuracy
* Validation accuracy
* Test accuracy
* Confusion matrix
* Loss curves / accuracy curves



---


## 🚀 Deployment Options

* Export to **TensorFlow Lite** for mobile apps
* Serve model through **Flask / FastAPI API**
* Integrate into a **mobile agriculture assistant**
* Use ONNX for cross-platform deployment

---

## 📦 requirements.txt (Example)

```
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
```
* **PlantVillage** for providing high-quality agricultural datasets
* **TensorFlow** for model development
* **EfficientNet** authors (Google AI
