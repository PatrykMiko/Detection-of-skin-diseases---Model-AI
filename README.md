# Detection-of-skin-diseases---Model-AI
The hybrid approach for multi- class skin lesion classification, combining the EfficientNetV2L architecture as a feature extractor with XGBoost as the classifier. The application of the Egret Swarm Optimization Algorithm for automatic hyperparameter fine-tuning of the XGBoost model.
# Skin Lesion Classification: A Hybrid Deep Learning & Evolutionary Ensemble Approach

This repository contains an end-to-end machine learning pipeline for the automated classification of skin diseases. The project leverages a hybrid architecture, combining the robust feature extraction capabilities of **EfficientNetV2L** with the high-performance classification of **XGBoost**, further optimized using the **Egret Swarm Optimization Algorithm (ESOA)**.

## 🚀 Technical Highlights & Engineering Choices

This project was built with a focus on performance, scalability, and state-of-the-art computer vision techniques:

* **Advanced Image Preprocessing:** Implemented custom OpenCV pipelines including **Dull Razor** (morphological Black-Hat transformations and telea inpainting) for automated hair removal, and **CLAHE** (Contrast Limited Adaptive Histogram Equalization) in the Lab color space to enhance lesion visibility without distorting color representations.
* **Performance-Optimized Data Pipelines:** Utilized `tf.data` with `prefetch` and `AUTOTUNE` to eliminate I/O bottlenecks during model training. Enabled **mixed-precision training** (`mixed_float16`) to maximize GPU memory efficiency and accelerate computation.
* **Two-Phase Transfer Learning:** Fine-tuned a pre-trained EfficientNetV2L (a highly efficient architecture developed by Google Research). Employed a warm-up phase for the custom top layers, followed by controlled unfreezing of the top 180 layers with a decayed learning rate to adapt to the specific medical dataset without catastrophic forgetting.
* **Hybrid CNN-Tree Architecture:** Extracted high-dimensional spatial features from the EfficientNetV2L's Global Average Pooling layer to feed into an XGBoost classifier, bridging the gap between deep representation learning and robust tabular ensemble methods. 
* **Evolutionary Hyperparameter Tuning:** Replaced standard grid/random search with MEALPY's **ESOA** (Egret Swarm Optimization Algorithm) to efficiently navigate the complex XGBoost hyperparameter space and minimize the classification error rate.

## 🛠️ Tech Stack

* **Deep Learning & ML:** TensorFlow 2.x, Keras, XGBoost, Scikit-learn
* **Computer Vision:** OpenCV (`cv2`), NumPy
* **Optimization:** MEALPY (Evolutionary Algorithms)
* **Data Handling & Utilities:** Kagglehub, OS, Shutil, Pickle
* **Visualization:** Matplotlib, Seaborn

## 🧠 Pipeline Architecture

### 1. Data Ingestion & Preprocessing (`preprocess.py`)
* Automatically fetches the `ahmedxc4/skin-ds` dataset via Kaggle's API.
* Processes images across train/val/test splits:
    1. Grayscale conversion & Morphological Black-Hat transformation.
    2. Gaussian Blurring & Binary Thresholding to create a hair mask.
    3. Inpainting (Dull Razor) to remove hair artifacts.
    4. Lanczos4 Interpolation resizing to 480x480 (or 224x224 for CNN ingestion).
    5. CLAHE application for localized contrast enhancement.

### 2. Feature Extraction & Deep Learning (`train.py`)
* Loads augmented datasets using `image_dataset_from_directory`.
* Constructs an `EfficientNetV2L` base model (ImageNet weights).
* **Phase 1:** Trains custom dense layers while the base model is frozen (Warm-up).
* **Phase 2:** Unfreezes the top 180 layers for fine-tuning using an `Adam(1e-5)` optimizer.
* Transforms the trained CNN into a headless feature extractor.

### 3. Classification & Optimization (`train.py`)
* Passes the un-augmented training set through the CNN to extract tabular feature vectors.
* Utilizes **ESOA** (population size = 15, epochs = 5) to optimize XGBoost hyperparameters (`max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`).
* Trains the final XGBoost model on the combined train + validation feature sets to maximize data utilization.

## 📊 Evaluation & Metrics

The model's performance is rigorously evaluated on a hold-out test set to ensure generalizability. Metrics include:
* **Accuracy & Classification Report:** Precision, Recall, and F1-score across all classes.
* **ROC-AUC:** Both Macro and Weighted averages to account for class imbalances.
* **Confusion Matrix:** Visualized via Seaborn to track true vs. predicted class distributions.

## 💻 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed and a CUDA-compatible GPU for accelerated training.

```bash
pip install tensorflow opencv-python numpy kagglehub xgboost mealpy scikit-learn matplotlib seaborn
