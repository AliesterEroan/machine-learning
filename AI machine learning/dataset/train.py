import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

if __name__ == "__main__":
    print("Loading extracted features...")
    # Load dataset splits
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")

    print("Training the Support Vector Machine (Linear Kernel)...")
    # Initialize and train SVM
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)

    # Predictions
    y_pred = svm_model.predict(X_test)

    # Metrics Calculations
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary') # Using binary since you have 2 objects

    print("\n" + "="*30)
    print("        MODEL METRICS")
    print("="*30)
    print(f"Accuracy Score : {accuracy * 100:.2f}%")
    print(f"F1 Score       : {f1:.4f}")
    print("="*30)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Object 1", "Object 2"]))

    # Save the trained model
    joblib.dump(svm_model, "model.joblib")
    print("Model successfully saved as 'model.joblib'!")