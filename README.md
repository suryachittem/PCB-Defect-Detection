# CircuitGuard ⚡ | PCB Defect Detection & Classification

CircuitGuard is an end-to-end automated system for detecting and classifying defects on **Printed Circuit Boards (PCBs)** using **image processing + deep learning**.  
It reduces the limitations of manual inspection by providing a fast, accurate, scalable defect inspection pipeline through a full-stack web application.

---

## 📌 Abstract
This project implements a robust PCB defect detection pipeline that combines:
- **Reference-based image subtraction** to localize defects between a template PCB and a test PCB.
- **ROI extraction using contours** for defect region cropping.
- **EfficientNet CNN classification** (**EfficientNet-B4**) to classify defects.
- **Full-stack integration** using a web-based UI + backend inference pipeline.

✅ Key results:
- **97%+ classification accuracy**
- **End-to-end processing ≤ 5 seconds per image**
- Exporting of **annotated images + CSV logs**

---

## 🎯 Project Objectives
- Detect and localize PCB defects by comparing defect-free template and test images.
- Extract defect ROIs using image processing techniques.
- Classify detected defects into predefined categories using **EfficientNet**.
- Provide a web-based interface to upload images and view annotated outputs.
- Export annotated outputs and prediction logs.

---

## 🏗️ System Architecture
CircuitGuard uses a 2-stage pipeline:

### 1️⃣ Subtraction Stage (Defect Localization)
- `cv2.absdiff()` → absolute difference between template and test images
- **Otsu Thresholding** → defect mask generation
- Noise reduction using morphological operations

### 2️⃣ Defect Extraction & Classification
- Morphological **erosion + dilation**
- **Contour detection**
- Bounding box extraction → defect ROI crops
- Each ROI is classified using **EfficientNet-B4 CNN model**

---

## 🔁 Workflow
1. Input:
   - Template PCB Image (defect-free)
   - Test PCB Image (may contain defects)
2. Image Subtraction (`absdiff`)
3. Thresholding (Otsu)
4. Morphological operations (Erode/Dilate)
5. Contour extraction
6. ROI extraction (cropping defects)
7. EfficientNet defect classification
8. Annotated output image + export logs

---

## 🧰 Technology Stack

### 🔹 Image Processing
- OpenCV
- NumPy

### 🔹 Model & Training
- PyTorch
- timm
- EfficientNet-B4
- Optimizer: Adam
- Loss: CrossEntropyLoss

### 🔹 Dataset
- DeepPCB Dataset

### 🔹 Frontend
- Streamlit / HTML + CSS + JavaScript

### 🔹 Backend
- Python
- Flask (Modularized inference pipeline)

### 🔹 Evaluation & Export
- Accuracy
- Loss
- Confusion Matrix
- CSV Logs
- Annotated Image export

---

## 📂 Project Structure
> (May vary based on your folder arrangement)

---bash
PCB_DATASET/
│── app.py
│── roi.py
│── requirements.txt
│── README.md
│── static/
│   ├── style.css
│   └── script.js
│── templates/
│   └── index.html
│── Efficient/
│   └── model_training.py
│── outputs/
│   └── annotated_results/
│── images/ (ignored in git)
│── Annotations/ (ignored in git)
│── train_images/ (ignored)
│── val_images/ (ignored)
│── test_images/ (ignored)
│── venv/ (ignored)

---bash

## ✅ Installation & Setup

### 1️⃣ Clone Repository

git clone https://github.com/suryachittem/PCB-Defect-Detection.git

cd PCB_DATASET

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python app.py

http://127.0.0.1:5000/

🧪 Model Training (EfficientNet)

python Efficient/model_training.py

📊 Evaluation

✅ System Evaluation Metrics

Defect mask accuracy

ROI localization performance

Classification accuracy ≥ 97%

Confusion matrix analysis

Upload-to-output time ≤ 5 seconds

📤 Outputs

CircuitGuard generates:

✅ Annotated PCB image with bounding boxes + labels

✅ CSV prediction logs

✅ Optional PDF export (if enabled)

📌 Results

EfficientNet classification accuracy: 97%+

Total processing time per image pair: ≤ 5 seconds

Robust detection of localized defect regions

🚀 Future Improvements

Add automatic template alignment / registration

Improve performance on complex background noise

Deploy with Docker + cloud inference

Mobile-friendly UI
