# 🤖 HOG + SVM Object Detection & Recognition

> Real-time object detection and recognition using HOG features and SVM classification! 🎯

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![Machine Learning](https://img.shields.io/badge/ML-HOG%2FSVM-orange?style=for-the-badge)

---

## ✨ Features

- 🎯 **Real-time object detection** using webcam
- 🧠 **HOG feature extraction** for robust object representation
- 🤖 **SVM classification** for accurate object recognition
- 📊 **Background subtraction** for motion detection
- 🔄 **Object tracking** with contour detection
- 📈 **Confidence scoring** for predictions
- 🎨 **Visual feedback** with bounding boxes and labels

## 🛠️ Requirements

### Software
- **Python 3.8+**
- **OpenCV 4.x**
- **scikit-learn**
- **numpy**
- **joblib**

### Hardware
- **Webcam** (USB or built-in)
- **Computer** with decent CPU for real-time processing

## 🔧 Installation

```bash
# Install required packages
pip install opencv-python scikit-learn numpy joblib

# Or use the requirements file
pip install -r requirement.txt
```

## 📋 Project Structure

```
hog-svm-object-detection/
├── dataset/
│   ├── object1/          # Training images for object 1
│   └── object2/          # Training images for object 2
├── model.joblib          # Trained SVM model
├── extract.txt           # Feature extraction script
├── train.txt             # Training script
├── vision.txt            # Real-time recognition script
└── requirement.txt      # Python dependencies
```

## 🚀 Getting Started

### 1. Prepare Your Dataset

**Collect training images:**
- Place images of object 1 in `dataset/object1/`
- Place images of object 2 in `dataset/object2/`
- Use 20+ images per object for better accuracy
- Ensure good lighting and varied angles

**Example dataset structure:**
```
dataset/
├── object1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── object2/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

### 2. Train the Model

```bash
# Navigate to project directory
cd hog-svm-object-detection

# Run the training script
python train.txt
```

**Training process:**
- Loads images from dataset folders
- Extracts HOG features (128x128 pixels)
- Trains SVM classifier with linear kernel
- Evaluates model on validation set
- Saves trained model as `model.joblib`

**Expected output:**
```
Chargement des données...
-> Chargement de 25 images depuis le dossier : object1
-> Chargement de 23 images depuis le dossier : object2
[OK] Données partitionnées : 38 exemples d'entraînement, 10 exemples de validation.
Entraînement du modèle SVM...
Évaluation sur les données de validation...
Précision sur les données de validation : 95.00%
Matrice de confusion :
[[5 0]
 [0 5]]
Sauvegarde du modèle...
[FÉLICITATIONS] Modèle sauvegardé avec succès sous le nom 'model.joblib'.
```

### 3. Run Real-time Recognition

```bash
# Run the vision/recognition script
python vision.txt
```

**Controls:**
- **`q`** - Quit the application
- **`c`** - Recalibrate background (if lighting changes)

**Features:**
- Automatic background calibration on startup
- Real-time object detection using motion detection
- SVM classification on detected objects
- Confidence percentage display
- Bounding box visualization

## 🔧 Configuration

### Camera Settings

**In vision.txt, modify camera parameters:**
```python
# Change camera index (0 = default, 1 = secondary)
cap = cv2.VideoCapture(1)

# Adjust resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### Detection Parameters

**Adjust sensitivity:**
```python
# Motion detection threshold
thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]

# Minimum contour area (noise filter)
if cv2.contourArea(largest_contour) > 2000:
```

### Prediction Interval

**Balance CPU vs responsiveness:**
```python
prediction_interval = 4  # Predict every 4 frames
```

## 🎯 How It Works

### 1. Feature Extraction (HOG)
- Converts images to grayscale
- Resizes to 128x128 pixels
- Extracts Histogram of Oriented Gradients
- Creates feature vectors for classification

### 2. Motion Detection
- Captures initial background frame
- Computes frame difference
- Applies Gaussian blur and thresholding
- Finds contours to detect moving objects

### 3. Object Recognition
- Extracts ROI (Region of Interest) from detected objects
- Resizes ROI to match training size
- Extracts HOG features
- Classifies using trained SVM model
- Displays confidence score

### 4. Visual Feedback
- Draws green bounding box around detected objects
- Shows object class label above box
- Displays confidence percentage
- Shows current status in corner

## 📊 Technical Details

**HOG Parameters:**
- Block size: 16x16
- Cell size: 8x8
- Block stride: 8x8
- Number of bins: 9

**SVM Configuration:**
- Kernel: Linear
- Probability: Enabled (for confidence scores)
- Multi-class: Binary classification

**Image Processing:**
- Input size: 128x128 pixels
- Color space: Grayscale
- Gaussian blur: 21x21 kernel

## 🎯 Typical Applications

- **Quality Control**: Object inspection on production lines
- **Surveillance**: Specific object detection in security systems
- **Robotics**: Object recognition for robotic arms
- **Education**: Computer vision teaching and demonstration
- **Research**: HOG+SVM methodology implementation

## ⚠️ Notes

- Ensure good lighting for reliable detection
- Training images should match test conditions
- Model performance depends on dataset quality
- CPU intensive - may lag on slower computers
- Camera calibration required for different environments

## 🎓 Learning Objectives

- Computer vision fundamentals
- Feature extraction techniques
- Machine learning classification
- Real-time image processing
- Motion detection algorithms

## 🔍 Troubleshooting

**Low accuracy:**
- Add more training images
- Ensure diverse angles and lighting
- Check image quality

**No detection:**
- Verify camera is working
- Check motion detection threshold
- Ensure objects are moving in frame

**Slow performance:**
- Increase prediction interval
- Reduce image resolution
- Close other applications

## 📈 Performance Metrics

**Expected results:**
- Training accuracy: 90-95%
- Real-time FPS: 15-30 (depending on hardware)
- Detection latency: 100-200ms
- Model size: ~1MB

---

**Made with ❤️ by AliesterEroan**