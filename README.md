# Machine Learning — HOG + SVM Object Recognition

> Real-time webcam recognition of two objects using HOG features and a linear SVM — a learning project, tested on the author's own captures.

Two-object computer-vision pipeline: images are captured per object, HOG (Histogram of Oriented Gradients) descriptors are extracted, a linear SVM is trained with scikit-learn, and a webcam script detects the largest moving blob via background subtraction and classifies it live. The repository contains two parallel working copies of the same experiment plus raw image assets.

---

## 📜 Table of Contents

- [✨ Features](#-features)
- [🚀 Installation](#-installation)
- [📖 Usage](#-usage)
- [🏗️ Architecture](#-architecture)
- [🛠️ Technologies Used](#-technologies-used)
- [⚠️ Limitations](#-limitations)
- [📜 License](#-license)
- [📧 Contact](#-contact)

---

## ✨ Features

### Pipeline
- **HOG feature extraction** — OpenCV `HOGDescriptor` on grayscale images; two variants exist in the repo: a default-descriptor version resizing to 128×128 (`train.txt`/`vision.txt`/`extract.txt`) and a custom 64×128 window version (`dataset/extract_features.py`/`dataset/vision.py`).
- **SVM classification** — `sklearn.svm.SVC` with a linear kernel, 80/20 stratified train/test split (`random_state=42`), saved with `joblib` (`model.joblib` / `svm_model.joblib`).
- **Metrics** — the `dataset/train.py` script reports accuracy, binary F1, and a classification report on the held-out test split.
- **Real-time webcam recognition** — `vision.py`/`vision.txt` calibrate a Gaussian-blurred background frame, find the largest contour above a 2000 px² area threshold, run SVM inference on the bounding-box ROI every 4th frame, and draw the label + probability on the live feed. Press `c` to recalibrate the background, `q` to quit.

### Repository contents
- **`AI machine learning/`** — main working folder: dataset folders (`dataset/object1`, `dataset/object2`, `dataset/test_images`), pipeline scripts, `SRC/OBJ 1` and `SRC/OBJ 2` raw captured frames, and generated artifacts (`model.joblib`, `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`).
- **`AI/`** — second copy of the dataset (`SRC/OBJ 1`, `SRC/OBJ 2`) with its own generated `.npy`/`.joblib` artifacts; no scripts of its own.
- **`Img source/`** — reference images and downloaded assets gathered during development (screenshots, icons, logos); not consumed by any script.

---

## 🚀 Installation

### Prerequisites
- Python 3.x
- A webcam (the recognition script opens `cv2.VideoCapture(1)`)

### Steps
1. Install dependencies: `pip install opencv-python numpy scikit-learn joblib`
   - Note: `AI machine learning/requirement.txt` and `AI machine learning/dataset/requierments.txt` exist but are empty placeholders — the list above is what the code actually imports.
2. Put training images into `AI machine learning/dataset/object1/` and `AI machine learning/dataset/object2/` (JPEG/PNG; the shipped `SRC/` folders contain the author's own `WIN_*.jpg` webcam captures if you want to reproduce the original run).
3. From `AI machine learning/dataset/`, run `python extract_features.py`, then `python train.py`, then `python vision.py`.
4. Alternatively, the `.txt` scripts in `AI machine learning/` contain the same pipeline as runnable Python — rename/copy them to `.py` and run from that folder with `dataset/train/objet1` and `dataset/train/objet2` populated.

---

## 📖 Usage

### Train a model
- `dataset/extract_features.py` reads images from a hardcoded path (`C:/Mohamed Mostfa/AI/dataset`, categories `object1`/`object2`), extracts HOG features, and writes `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`.
- `dataset/train.py` trains `SVC(kernel='linear', C=1.0)` on those splits, prints accuracy/F1/classification report, and saves `model.joblib`.

### Recognize objects live
- `dataset/vision.py` loads `model.joblib` and runs the webcam loop described above. Labels are hardcoded to `"Objet 1"` / `"Objet 2"`.

---

## 🏗️ Architecture

```
machine learning/
├── AI machine learning/
│   ├── dataset/
│   │   ├── object1/, object2/   # per-class training images
│   │   ├── test_images/         # held-out test images
│   │   ├── extract_features.py  # HOG extraction → .npy splits
│   │   ├── train.py             # SVM training → model.joblib
│   │   ├── vision.py            # webcam recognition loop
│   │   └── requierments.txt     # empty placeholder
│   ├── SRC/OBJ 1, OBJ 2/        # raw captured frames per object
│   ├── train.txt / vision.txt / extract.txt   # same pipeline as .py code in .txt files
│   ├── model.joblib + *.npy     # trained artifacts
│   └── read me.txt              # original French project notes
├── AI/                 # duplicate dataset + artifacts (model.joblib, .npy)
├── Img source/         # reference images/assets (not used by scripts)
└── .gitignore
```

### Data Flow
1. Images per object class → grayscale → resize → HOG descriptor → feature vector.
2. Vectors + labels → 80/20 stratified split → linear SVM → `model.joblib`.
3. Webcam frame → background subtraction → largest contour → ROI → same HOG extraction → SVM predict → on-screen label.

---

## 🛠️ Technologies Used

### Core
- **Python** - all pipeline scripts
- **OpenCV (`cv2`)** - image loading, HOG descriptors, webcam capture, contour tracking, overlay drawing
- **scikit-learn** - `SVC` classifier, `train_test_split`, `accuracy_score`, `f1_score`, `classification_report`
- **NumPy / joblib** - feature arrays (`.npy`) and model serialization (`.joblib`)

---

## ⚠️ Limitations

### Algorithm / Model
- **Two classes only** — class labels are hardcoded to `object1`/`object2` (or `objet1`/`objet2` in the `.txt` variants); anything else is out of scope.
- **Two different HOG configurations coexist** — the `.txt`/`dataset/vision.py` scripts must match the descriptor parameters used at training time or features won't align; the repo ships both variants without a single canonical entry point.
- **Background-subtraction tracking** detects the single largest moving contour above a fixed 2000 px² threshold — it cannot handle multiple simultaneous objects, heavy occlusion, or a moving camera/background.

### Technical
- **Hardcoded dataset path** — `dataset/extract_features.py` points to `C:/Mohamed Mostfa/AI/dataset`, which does not exist on a fresh clone; edit `DATASET_PATH` before running.
- **Webcam index fixed** — `vision.py`/`vision.txt` open `cv2.VideoCapture(1)`; adjust if your camera enumerates differently.
- **Scripts-as-txt** — `train.txt`, `vision.txt`, `extract.txt`, `read me.txt`, and both `requirement(s).txt` files contain code/docs but are not runnable under their current names; requirements files are empty.
- **Duplicate folders** — `AI/` and `AI machine learning/` overlap heavily (same images, separate artifacts); it is not obvious which artifacts correspond to which script variant.

### Testing Limitations
- **What a clone can re-check**: scripts are syntactically readable Python; you can retrain on your own `object1`/`object2` images after fixing `DATASET_PATH`.
- **What a clone cannot re-check**: the shipped `model.joblib`/`.npy` artifacts were produced on the author's machine against the author's captures; no accuracy numbers are recorded in the repo, and the webcam demo requires physical hardware.

---

## 📜 License

Machine Learning is proprietary software — see [LICENSE](LICENSE) for the full terms.

**Copyright © 2026 AliesterEroan. All rights reserved.**

This software is provided for personal, non-commercial use only. Commercial use, redistribution, modification, and sublicensing are prohibited without explicit written permission. The copyright holder may distribute copies through the project's official GitHub Releases page; third parties may not host, mirror, or redistribute this software.

For licensing inquiries, open an issue at <https://github.com/AliesterEroan/machine-learning/issues>.

---

## 📧 Contact

- **GitHub**: [AliesterEroan/machine-learning](https://github.com/AliesterEroan/machine-learning)
- **Issues**: [Create an issue](https://github.com/AliesterEroan/machine-learning/issues)

---

*Built with ❤️ by AliesterEroan*
