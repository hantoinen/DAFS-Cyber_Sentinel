# version plus robuste"
import json
import os
from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_recall_curve, auc
import numpy as np

def evaluate(model, X_test, y_test, output_dir="outputs", threshold=0.3):
    """
    Évalue le modèle avec un seuil personnalisé pour maximiser le Recall
    et sauvegarde les métriques au format JSON.
    """
    # 1. Prédire les probabilités (nécessaire pour changer le seuil et l'AUC-PR)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 2. Appliquer le seuil personnalisé (ex: 0.3 au lieu de 0.5)
    # Si la probabilité est > threshold, on prédit 1 (anomalie), sinon 0
    y_pred = (y_proba >= threshold).astype(int)
    
    # 3. Calculer les métriques demandées
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Calcul de l'AUC-PR (Area Under Precision-Recall Curve)
    precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recalls, precisions)
    
    # Matrice de confusion : [[Vrais Négatifs, Faux Positifs], [Faux Négatifs, Vrais Positifs]]
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist() # Converti en liste pour le JSON
    }
    
    # 4. Sauvegarde dans le dossier outputs
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"--- Évaluation Terminée (Seuil: {threshold}) ---")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | PR-AUC: {pr_auc:.4f}")
    
    return metrics