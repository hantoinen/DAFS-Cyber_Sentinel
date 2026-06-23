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
    y_pred = model.predict(X_test)
    if hasattr(model.named_steps["model"], "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback: some estimators only have decision_function
        if hasattr(model.named_steps["model"], "decision_function"):
            scores = model.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            y_proba = y_pred.astype(float)

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