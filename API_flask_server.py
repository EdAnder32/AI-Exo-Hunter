## An python API for our Model
## Design by edander32 (Edmilson Alexandre) and jjambo(Joaquim Jambo)

from flask import Flask, request, jsonify
import pandas as pd
import joblib

import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("exoplanet_model.pkl")

# Número total de features do modelo
NUM_FEATURES = model.n_features_in_

# Índices das suas features importantes no array de treino
IMPORTANT_FEATURES = {
    "temperature": 0,
    "stellar_temp": 1,
    "depth": 2,
    "stellar_gravity": 3,
    "duration": 4,
    "stellar_radius": 5,
    "magnitude": 6,
    "insol": 7,
    "period": 8,
    "planet_radius": 9
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Inicializa array com zeros
        row = [0] * NUM_FEATURES
        
        # Preenche só as features importantes
        for key, idx in IMPORTANT_FEATURES.items():
            if key in data:
                row[idx] = data[key]
        
        df = pd.DataFrame([row])
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]
        return jsonify({"prediction": int(pred), "probability": float(prob)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)


