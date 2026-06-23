import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

CATEGORICAL = ["protocol_type", "service", "flag"]
TARGET = "class"

def preprocess(df: pd.DataFrame) -> Tuple:
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
