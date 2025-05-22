from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")

preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
model = joblib.load(os.path.join(MODELS_DIR, "svm.joblib"))


species_map = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    X = preprocessor.transform(df)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()
    return jsonify({"species": species_map[pred], "probability": float(prob)})

if __name__ == "__main__":
    app.run(port=5001)
