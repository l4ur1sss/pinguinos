from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
from penguins_classifier.preprocessing import load_and_prepare_data

MODELS_DIR = "models"

def train_and_save_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare_data()

    # Guardar preprocesador
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))

    models = {
        "logistic_regression": LogisticRegression(max_iter=200),
        "svm": SVC(probability=True),
        "decision_tree": DecisionTreeClassifier(),
        "knn": KNeighborsClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))
        print(f"Modelo {name} entrenado y guardado.")

if __name__ == "__main__":
    train_and_save_models()
