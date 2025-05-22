from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
model = joblib.load(os.path.join(MODELS_DIR, "decision_tree.joblib"))

species_map = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    X = preprocessor.transform(df)
    pred = model.predict(X)[0]
    # DecisionTree no tiene predict_proba por defecto (sí en sklearn 1.0+), si falla no enviamos prob
    try:
        prob = model.predict_proba(X)[0].max()
        return jsonify({"species": species_map[pred], "probability": float(prob)})
    except AttributeError:
        return jsonify({"species": species_map[pred]})

if __name__ == "__main__":
    app.run(port=5002)
