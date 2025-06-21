# Plant Disease Classification Using Deep Learning

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model Architectures](#model-architectures)
  - [Simple CNN](#simple-cnn)
  - [EfficientNetB0](#efficientnetb0)
- [Installation](#installation)
- [Usage](#usage)
  - [Environment Setup](#environment-setup)
  - [Data Preparation](#data-preparation)
  - [Training Models](#training-models)
  - [Running the Web App](#running-the-web-app)
- [Results & Comparison](#results--comparison)
- [Project Structure](#project-structure)
- [Contribution](#contribution)
- [License](#license)

---

## Project Overview

This repository contains code and resources for classifying apple leaf diseases using deep learning. Two models are trained and compared:

1. **Simple CNN**: A lightweight Convolutional Neural Network built from scratch.
2. **EfficientNetB0**: A transfer learning approach leveraging a pre-trained EfficientNetB0.

Both models are evaluated on an augmented dataset of 12,000 images across four classes:

- Apple Scab
- Black Rot
- Cedar Apple Rust
- Healthy

A Streamlit web application demonstrates inference.
## Dataset

The augmented Apple Disease Detection dataset is available on Kaggle: [Augmented Apple Disease Detection Dataset](https://www.kaggle.com/datasets/rm1000/augmented-apple-disease-detection-dataset)

Each class contains 3,000 images, covering various conditions, angles, and lighting to improve model robustness.

## Features

- Two deep learning models (Simple CNN & EfficientNetB0)
- Data splitting into train/validation/test
- Training scripts with callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- Performance comparison metrics (accuracy, precision, recall, F1-score)
- Streamlit web app with upload interface and Grad-CAM heatmaps

## Model Architectures

### Simple CNN

- Input: 224×224×3 RGB images
- Layers:
  - Conv2D(32) → ReLU → MaxPooling(2)
  - Conv2D(64) → ReLU → MaxPooling(2)
  - Flatten → Dense(256) → ReLU → Dense(4) → Softmax
- \~500K parameters
- Trained for 5 epochs with Adam optimizer

### EfficientNetB0

- Base: Pre-trained EfficientNetB0 (ImageNet)
- Input: 256×256×3 RGB images
- Custom head:
  - Global Average Pooling
  - BatchNormalization → Dropout(0.5)
  - Dense(256) → ReLU → BatchNormalization → Dropout(0.3)
  - Dense(4) → Softmax
- \~5.3M parameters
- Two-phase training:
  1. Feature extraction (base frozen)
  2. Fine-tuning (top layers unfrozen)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/plant-disease-classification.git
   cd plant-disease-classification
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # on Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Environment Setup

Ensure you have Python 3.8+ and GPU support (optional but recommended).

### Data Preparation

1. Download and unzip the Kaggle dataset into `data/train_augmented/`.
2. Run the data splitting script to organize train/validation/test sets:
   ```bash
   python scripts/split_dataset.py
   ```

### Training Models

#### Simple CNN

```bash
python scripts/train_simple_cnn.py
```

#### EfficientNetB0

```bash
python scripts/train_efficientnetb0.py
```

Model weights and training logs will be saved to `models/` and `logs/fit/` respectively.

### Running the Web App

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the displayed URL in your browser to upload leaf images, view predictions, and Grad-CAM visualizations.

## Results & Comparison

Refer to the [Comprehensive Model Comparison Report](comprehensive_model_comparison_report.md) for detailed metrics:

- EfficientNetB0 achieves ≈97.8% test accuracy vs. Simple CNN’s 90%
- Trade-offs in inference time, model size, and parameter count

## Project Structure

```
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── scripts/
│  
│   ├── train_simple_cnn.ipynb        # Simple CNN training
│   └── train_efficientnetb0.ipynb    # EfficientNetB0 training
├── models/
│   ├── simple_cnn.h5              # Trained Simple CNN model   
│   └── apple_disease_model_final.h5 # Final combined model
├── data/
│   └── train_augmented/           # Downloaded & augmented dataset
├── class_indices.json             # JSON mapping of classes
├── comprehensive_model_comparison_report.md
├── presentation.pptx              # Project presentation
└── README.md
```

## Contribution

Contributions are welcome! Please open issues or submit pull requests to improve functionality, add new models, or optimize performance.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

