import requests
import json

SERVERS = {
    "logistic_regression": "http://127.0.0.1:5000/predict",
    "svm": "http://127.0.0.1:5001/predict",
    "decision_tree": "http://127.0.0.1:5002/predict",
    "knn": "http://127.0.0.1:5003/predict"
}

# Ejemplo de datos de pingüino (ya preprocesados o con las variables que el servidor espera)
sample_input = {
    "bill_length_mm": 43.2,
    "bill_depth_mm": 17.1,
    "flipper_length_mm": 210,
    "body_mass_g": 4500,
    "island": "Biscoe",
    "sex": "Male"
}

def test_servers():
    for name, url in SERVERS.items():
        print(f"Probando servidor {name}...")
        for i in range(2):  # Dos peticiones por modelo
            response = requests.post(url, json=sample_input)
            print(f"Respuesta {i+1}: {response.json()}")

if __name__ == "__main__":
    test_servers()
