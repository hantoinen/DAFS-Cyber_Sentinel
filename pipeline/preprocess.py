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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    Construit le ColumnTransformer adapté au dataset réseau.

    Numériques  → imputation médiane  + StandardScaler
    Catégoriels → imputation mode     + OneHotEncoder
    """
    numeric_cols = _get_numeric_cols(X)
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",   # ignore toute colonne inattendue
    )


def preprocess(df: pd.DataFrame):
    """
    Nettoie, sépare et prépare le dataset pour l'entraînement.

    Retourne
    --------
    X_train, X_test : pd.DataFrame  — features brutes (non transformées)
    y_train, y_test : pd.Series     — cible binaire 0/1
    preproc         : ColumnTransformer non entraîné
    """
    df = df.copy()

    # 1. Supprimer les colonnes inutiles
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 2. Vérifications de base
    if TARGET_COL not in df.columns:
        raise KeyError(
            f"Colonne cible '{TARGET_COL}' absente. "
            f"Colonnes disponibles : {list(df.columns)}"
        )
    missing_cats = [c for c in CATEGORICAL_COLS if c not in df.columns]
    if missing_cats:
        raise ValueError(f"Colonnes catégorielles manquantes : {missing_cats}")

    # 3. Supprimer les lignes sans cible
    df = df.dropna(subset=[TARGET_COL])

    # 4. Séparer features / cible
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # 5. Split train / test (stratifié → garde le ratio normal/anomaly)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 6. Préprocesseur (non entraîné — sera fit dans train_model)
    preproc = build_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preproc