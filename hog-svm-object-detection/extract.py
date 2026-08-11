import cv2
import os
import numpy as np

def extract_hog_features(image):
    # Changement ici également pour être aligné avec le modèle d'entraînement
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    features = hog.compute(image)
    if features is None:
        return None
    return features.flatten()

def load_dataset(dataset_path, image_size=(128, 128)):
    X, y = [], []
    class_labels = {"objet1": 0, "objet2": 1}
    
    for label, class_id in class_labels.items():
        folder = os.path.join(dataset_path, label)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Le dossier {folder} est introuvable. Vérifiez le chemin.")

        for filename in os.listdir(folder):
            image_path = os.path.join(folder, filename)
            if not os.path.isfile(image_path):
                continue
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Erreur lors du chargement de l'image : {image_path}")
                continue

            # Redimensionner et convertir en caractéristiques HOG
            image = cv2.resize(image, image_size)
            features = extract_hog_features(image)
            if features is not None:
                X.append(features)
                y.append(class_id)
    
    return np.array(X), np.array(y)

# Utilisation du chemin relatif universel
dataset_path = "./dataset/train"

if not os.path.exists(dataset_path):
    print(f"Erreur : Le chemin {dataset_path} n'existe pas.")
else:
    print(f"Chemin valide trouve : {dataset_path}")
    X, y = load_dataset(dataset_path)
    print(f"Verification reussie : {X.shape[0]} exemples trouves avec un vecteur de taille {X.shape[1]}.")