import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Fonction pour extraire les caractéristiques HOG
def extract_hog_features(image):
    # Conversion systématique en niveaux de gris pour garantir la cohérence des dimensions
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    features = hog.compute(image)
    if features is None:
        return None
    return features.flatten()

# Chargement et étiquetage des images
def load_dataset(dataset_path, image_size=(128, 128)):
    X, y = [], []
    # Configuration stricte pour vos DEUX objets
    class_labels = {"objet1": 0, "objet2": 1}

    for label, class_id in class_labels.items():
        folder = os.path.join(dataset_path, label)
        
        if not os.path.exists(folder):
            raise FileNotFoundError(f"[ERREUR CRITIQUE] Le dossier '{folder}' n'existe pas. Veuillez le créer.")

        images_in_folder = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        if len(images_in_folder) == 0:
            raise ValueError(f"[ERREUR CRITIQUE] Le dossier '{folder}' est VIDE. Ajoutez-y des images.")

        print(f"-> Chargement de {len(images_in_folder)} images depuis le dossier : {label}")
        
        for filename in images_in_folder:
            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue

            image = cv2.resize(image, image_size)
            features = extract_hog_features(image)
            
            if features is not None:
                X.append(features)
                y.append(class_id)
    
    return np.array(X), np.array(y)

# Chemin du dataset
dataset_path = "./dataset/train"

try:
    print("Chargement des données...")
    X, y = load_dataset(dataset_path)

    # Vérification du nombre de classes chargées au total
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(f"[ERREUR] Le dataset global ne contient qu'une seule classe ({unique_classes}). Il en faut 2.")

    # Utilisation de stratify=y pour équilibrer les classes lors du split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"[OK] Données partitionnées : {X_train.shape[0]} exemples d'entraînement, {X_val.shape[0]} exemples de validation.")

    # Entraînement du SVM
    print("Entraînement du modèle SVM...")
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    # Évaluation
    print("Évaluation sur les données de validation...")
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Précision sur les données de validation : {accuracy * 100:.2f}%")

    # Matrice de confusion
    cm = confusion_matrix(y_val, y_pred)
    print("Matrice de confusion :\n", cm)

    # Sauvegarde locale automatique
    print("Sauvegarde du modèle...")
    joblib.dump(model, "svm_model.joblib")
    print("[FÉLICITATIONS] Modèle sauvegardé avec succès sous le nom 'svm_model.joblib'.")

except Exception as e:
    print(f"\n--- INTERRUPTION DU SCRIPT ---")
    print(e)