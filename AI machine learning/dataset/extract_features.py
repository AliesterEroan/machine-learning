import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define paths
DATASET_PATH = "C:/Mohamed Mostfa/AI/dataset"
CATEGORIES = ["object1", "object2"]
IMG_SIZE = (64, 128)  # HOG works incredibly well with 1:2 aspect ratios (like 64x128)

def extract_hog_features(img):
    """Extracts Histogram of Oriented Gradients (HOG) features from an image."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize image
    gray_resized = cv2.resize(gray, IMG_SIZE)
    
    # Initialize HOG descriptor
    hog = cv2.HOGDescriptor(_winSize=(64, 128),
                            _blockSize=(16, 16),
                            _blockStride=(8, 8),
                            _cellSize=(8, 8),
                            _nbins=9)
    
    # Compute HOG features
    hog_features = hog.compute(gray_resized)
    return hog_features.flatten()

if __name__ == "__main__":
    data = []
    labels = []

    print("Extracting features from dataset...")
    for label_idx, category in enumerate(CATEGORIES):
        dir_path = os.path.join(DATASET_PATH, category)
        if not os.path.exists(dir_path):
            print(f"Warning: Path not found {dir_path}")
            continue
            
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue  # Skip corrupt files
                
            try:
                features = extract_hog_features(img)
                data.append(features)
                labels.append(label_idx)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    X = np.array(data)
    y = np.array(labels)

    # 80% Training, 20% Testing split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Save the processed data splits
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    print(f"Feature extraction complete! Saved splits to disk. Total samples: {len(X)}")