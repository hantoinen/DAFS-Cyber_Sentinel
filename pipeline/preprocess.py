"""
pipeline/preprocess.py

Responsable : preprocessing
Dataset     : Intrusion Detection (KDD-style)
              colonnes catégorielles : protocol_type, service, flag
              cible                  : class (0 = normal, 1 = anomaly)

Contrat imposé par main.py — NE PAS changer la signature :
    preprocess(df) -> (X_train, X_test, y_train, y_test, preproc)

Note : preproc est renvoyé NON entraîné.
       C'est train_model qui fait pipe.fit(X_train, y_train).
       → Pas de data leakage du test vers le train.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ── Constantes (à ajuster si le CSV change) ──────────────────────────────────
TARGET_COL = "class"
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
DROP_COLS: list[str] = []   # ex. colonnes id ou fuite de données
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ─────────────────────────────────────────────────────────────────────────────


def _get_numeric_cols(X: pd.DataFrame) -> list[str]:
    """Toutes les colonnes numériques (hors catégorielles déjà listées)."""
    return [
        c for c in X.select_dtypes(include="number").columns
        if c not in CATEGORICAL_COLS
    ]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Split into train/test and build a robust preprocessing pipeline.
    Includes imputation, scaling, and OneHotEncoding via ColumnTransformer.
    Returns: X_train, X_test, y_train, y_test, preprocessor
    """
    # 1. Séparation de la cible et des features
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    # 2. Détection automatique des types de colonnes
    num_cols = [c for c in X.columns if c not in CATEGORICAL]
    cat_cols = [c for c in CATEGORICAL if c in X.columns]

    # 3. Création des sous-pipelines pour chaque type de données
    
    # Pipeline Numérique : Imputation (médiane) + Normalisation Robuste (anti-outliers)
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(with_centering=False)) # with_centering=False conserve la compatibilité sparse
    ])

    # Pipeline Catégoriel : Imputation (valeur la plus fréquente) + One-Hot Encoding
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    # 4. Assemblage global avec ColumnTransformer
    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # 5. Split Train/Test avec stratification pour respecter la distribution des classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, preproc
