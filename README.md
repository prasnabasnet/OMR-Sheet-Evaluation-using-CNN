

#  OMR Sheet Evaluation System

> An automated Optical Mark Recognition (OMR) pipeline that detects, classifies, and scores bubble answers on exam sheets using **Computer Vision** and a **Convolutional Neural Network (CNN)**.

---

##Result of the GUI

| OMR Sheet Input | Evaluator UI Output |
|---|---|
| !![Image Alt](https://github.com/prasnabasnet/OMR-Sheet-Evaluation-using-CNN/blob/main/omrdataset.jpg?raw=true) | ![Image Alt](https://github.com/prasnabasnet/OMR-Sheet-Evaluation-using-CNN/blob/main/result.jpg?raw=true) |

---

##  Overview

This project automates the evaluation of OMR (bubble sheet) answer papers. Instead of manual checking, the system:

1. **Detects** all bubbles on a scanned OMR sheet using OpenCV
2. **Classifies** each bubble as `filled`, `empty`, or `invalid` using a trained CNN model
3. **Scores** the paper and displays results in a GUI interface

---

##  CNN Architecture

The model is a multi-layer CNN trained to classify individual bubble crops into 3 classes.

!![Image Alt](https://github.com/prasnabasnet/OMR-Sheet-Evaluation-using-CNN/blob/main/cnn_architecture.png?raw=true)

| Layer | Details |
|---|---|
| Input | 64×64 grayscale image |
| Conv Block 1 | Conv2D × 2 + MaxPooling |
| Conv Block 2 | Conv2D × 2 + MaxPooling |
| Dense | Fully connected + Dropout |
| Output | 3 classes: `filled`, `empty`, `invalid` |

Hyperparameters were tuned automatically using **Keras Tuner (RandomSearch)** across 5 trials, optimizing for `val_accuracy`.

---

## Image Preprocessing

### Bubble Detection with Otsu's Thresholding

![Image Alt](https://github.com/prasnabasnet/OMR-Sheet-Evaluation-using-CNN/blob/main/otsuthreshold.png?raw=true)

Each OMR sheet is preprocessed through:
- **Skew Correction** — Rotates the sheet to align it properly
- **Grayscale Conversion** — Removes color noise
- **Gaussian Blur** — Smooths out minor artifacts
- **Adaptive Thresholding** — Converts to binary image for contour detection
- **Contour Filtering** — Filters by area (400–2500 px²), size (20–60 px), and aspect ratio (0.8–1.2) to isolate bubbles

---

## Results

### Training & Validation Accuracy

![Image Alt](https://github.com/prasnabasnet/OMR-Sheet-Evaluation-using-CNN/blob/main/T&A.jpg?raw=true)

The model converges quickly and achieves **~99% validation accuracy** within a few epochs.

### Training & Validation Loss

![Image Alt](https://github.com/prasnabasnet/OMR-Sheet-Evaluation-using-CNN/blob/main/t&vloss.jpg?raw=true)

Training loss drops sharply and validation loss remains consistently low, indicating no significant overfitting.

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

| Class | Correct | Misclassified |
|---|---|---|
| empty | 95 / 95 | 0 |
| filled | 235 / 237 | 2 |
| invalid | 199 / 200 | 1 |

**Overall Test Accuracy: ~99.4%**

### Softmax Classification Boundary

![Softmax Classification](images/softmax_classification.png)

The softmax output layer cleanly separates the 3 bubble classes in feature space.

---

## 🖥️ GUI Interface

The desktop GUI allows teachers to:
- Upload any scanned OMR sheet
- View the detected answer per question (A/B/C/D, Invalid, or Unfilled)
- See the final score automatically calculated

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Deep Learning | TensorFlow, Keras |
| Hyperparameter Tuning | Keras Tuner |
| Computer Vision | OpenCV |
| Data Processing | NumPy, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| GUI | Tkinter |
| Environment | Google Colab + Google Drive |

---

## Project Structure

```
omr-evaluation-system/
│
├── bubble_extraction.py       # Stage 1: Extract bubbles from OMR sheets
├── dataset_split.py           # Stage 2: Split dataset into train/val/test
├── cnn_training.py            # Stage 3: CNN model training with Keras Tuner
├── gui_evaluator.py           # Stage 4: Desktop GUI for evaluation
│
├── omr_dataset/
│   ├── train/
│   │   ├── filled/
│   │   ├── empty/
│   │   └── invalid/
│   ├── val/
│   └── test/
│
├── best_cnn_model.h5          # Saved trained model
└── README.md
```


