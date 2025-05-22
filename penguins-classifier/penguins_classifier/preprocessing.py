import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def load_and_prepare_data():
    # Carga y limpieza del dataset
    df = sns.load_dataset("penguins").dropna()

    # Variables objetivo y características
    X = df.drop("species", axis=1)
    y = df["species"]

    # Convertir y a valores numéricos
    y_num = y.map({"Adelie":0, "Chinstrap":1, "Gentoo":2}).values

    # Columnas categóricas y numéricas
    categorical_features = ['island', 'sex']
    numeric_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

    # Pipeline para procesar variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_num, test_size=0.2, random_state=42, stratify=y_num)

    # Ajustar y transformar entrenamiento, transformar test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
