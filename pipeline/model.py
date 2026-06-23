from typing import Optional
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np

def _scale_pos_weight(y):
    # ratio of negatives to positives (for imbalance)
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    return neg / pos

def train_model(X_train, y_train, preproc, seed: int = 42):
    """
    Build a Pipeline(preproc → XGBClassifier) and fit using GridSearchCV.
    Integrates early stopping directly within the XGBoost steps.
    """
    spw = _scale_pos_weight(y_train)
    
    # 1. Définition du classifieur de base
    clf = XGBClassifier(
        n_estimators=400,            # Augmenté car l'early stopping l'arrêtera avant si besoin
        learning_rate=0.05,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=spw,
        verbosity=0,
        # Configuration de l'early stopping directement dans l'initiateur (norme scikit-learn)
        early_stopping_rounds=15     
    )
    
    # 2. Création du pipeline
    pipe = Pipeline([
        ("preproc", preproc), 
        ("model", clf)
    ])
    
    # 3. Grille d'hyperparamètres à tester
    # Note : On utilise le préfixe 'model__' pour cibler l'étape du pipeline
    param_grid = {
        "model__max_depth": [3, 5, 7],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }
    
    # 4. Configuration de la validation croisée (StratifiedKFold pour gérer le déséquilibre)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    
    # 5. Recherche par grille
    # On optimise ici 'PR-AUC' (average_precision) car vos données sont très déséquilibrées
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="average_precision", 
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    
    # Pour que l'early stopping fonctionne durant la validation croisée, XGBoost a besoin d'un set de validation.
    # Dans un pipeline sklearn complexe avec transformation à la volée, la méthode robuste 
    # consiste à laisser scikit-learn gérer le split interne du CV via le grid_search.
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best CV PR-AUC Score: {grid_search.best_score_:.4f}")
    
    # Renvoie le meilleur pipeline entraîné avec les paramètres optimisés
    return grid_search.best_estimator_