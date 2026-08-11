#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import joblib
import numpy as np
import sys

# Fonction pour extraire les caractéristiques HOG
def extract_hog_features(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    features = hog.compute(image)
    if features is None:
        return None
    return features.flatten()

# Charger le modèle entraîné
print("[INFO] Chargement du modele...")
try:
    model = joblib.load("svm_model.joblib")
    print("[OK] Modele charge avec succes")
except FileNotFoundError:
    print("[ERREUR] Fichier 'svm_model.joblib' introuvable dans le dossier.")
    sys.exit(1)

# Dictionnaire de correspondance pour vos deux objets
class_names = {0: "Objet 1", 1: "Objet 2"}

# Initialisation de la webcam
print("[INFO] Ouverture de la webcam...")
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("[ERREUR] Impossible d'acceder a la webcam")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Laissez la vue vide 1 seconde pour calibrer le fond.")
print("[INFO] Touches : 'q' pour quitter, 'c' pour recalibrer le fond si la lumiere change.")
print("=" * 60)

# Mémorisation du fond initial pour le tracking physique
ret, initial_frame = cap.read()
gray_background = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
gray_background = cv2.GaussianBlur(gray_background, (21, 21), 0)

frame_count = 0
prediction_interval = 4  # Ajustement pour la réactivité CPU
current_label = "Aucun objet"
confidence_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] Capture d'image echouee")
        break
    
    frame_count += 1
    
    # Prétraitement pour le suivi de contours
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    
    # Différence absolue avec le fond mémorisé
    frame_delta = cv2.absdiff(gray_background, gray_frame)
    thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Recherche de la présence d'un objet
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_detected = False
    
    if contours:
        # On cible le contour le plus imposant à l'écran
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 2000:  # Seuil anti-bruit de pixels
            object_detected = True
            (x, y, w, h) = cv2.boundingRect(largest_contour)
            
            # Étape d'inférence SVM sur la zone d'intérêt (ROI)
            if frame_count % prediction_interval == 0:
                try:
                    roi = frame[y:y+h, x:x+w]
                    if roi.size > 0:
                        roi_resized = cv2.resize(roi, (128, 128))
                        features = extract_hog_features(roi_resized)
                        
                        if features is not None:
                            features = features.reshape(1, -1)
                            class_id = model.predict(features)[0]
                            current_label = class_names.get(class_id, "Inconnu")
                            
                            if hasattr(model, "predict_proba"):
                                probs = model.predict_proba(features)[0]
                                confidence_text = f" ({probs[class_id]*100:.0f}%)"
                except Exception as e:
                    print(f"[ERREUR] Échec de la prédiction : {e}")

            # Dessiner le rectangle vert en temps réel autour de l'objet
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Écrire la classe juste au-dessus du cadre vert
            text = f"{current_label}{confidence_text}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if not object_detected:
        current_label = "Aucun objet"
        confidence_text = ""

    # Incrustation du bandeau d'état général
    cv2.putText(frame, f"Statut: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Reconnaissance d'objets (2 objets + Tracking)", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):  # Recalibrage manuel si déplacement ou changement d'éclairage
        gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_background = cv2.GaussianBlur(gray_background, (21, 21), 0)
        print("[INFO] Fond recalibre.")

cap.release()
cv2.destroyAllWindows()
print("[INFO] Programme termine")