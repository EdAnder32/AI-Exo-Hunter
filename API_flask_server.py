from flask import Flask, request, jsonify
from flask_cors import cross_origin
import pandas as pd
import joblib
import math

app = Flask(__name__)

# ======== MODELO .PKL ========
model = joblib.load("exoplanet_model.pkl")

NUM_FEATURES = model.n_features_in_
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


# ======== ROTA DE PREDIÇÃO ========
@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    try:
        data = request.get_json()
        row = [0] * NUM_FEATURES
        for key, idx in IMPORTANT_FEATURES.items():
            if key in data:
                row[idx] = data[key]

        df = pd.DataFrame([row])
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]
        return jsonify({"prediction": int(pred), "probability": float(prob)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ======== ROTA DE PLANETAS CONFIRMADOS ========
@app.route("/confirmed", methods=["GET"])
@cross_origin()
def confirmed():
    try:
        # Lê o arquivo CSV com os dados
        df = pd.read_csv("content.csv")

        # Filtra apenas exoplanetas confirmados
        confirmed_planets = df[df["koi_disposition"] == "CONFIRMED"].copy()

        # Remove planetas sem nome conhecido
        confirmed_planets = confirmed_planets.dropna(subset=["kepler_name"])

        # Limita a 20 resultados
        confirmed_planets = confirmed_planets.head(20)

        # Função auxiliar para classificar o tamanho
        def classify_size(r_earth):
            if pd.isna(r_earth):
                return "Unknown"
            if r_earth < 0.8:
                return "Sub-Earth"
            elif r_earth < 1.2:
                return "Earth-size"
            elif r_earth < 2.0:
                return "Super-Earth"
            elif r_earth < 6.0:
                return "Neptune-size"
            else:
                return "Jupiter-size"

        # Função auxiliar para classificar o tipo da estrela
        def classify_star(temp):
            if pd.isna(temp):
                return "Unknown"
            if temp < 3700:
                return "Red Dwarf (M)"
            elif temp < 5200:
                return "K-type Star (Orange Dwarf)"
            elif temp < 6000:
                return "G-type Star (Sun-like)"
            elif temp < 7500:
                return "F-type Star (White)"
            elif temp < 10000:
                return "A-type Star (Bluish-white)"
            else:
                return "Hot Blue Star"

        # Monta o JSON de retorno
        planets = []
        for _, row in confirmed_planets.iterrows():
            name = row["kepler_name"]
            size = classify_size(row.get("koi_prad", math.nan))
            period = f"{row.get('koi_period', 'Unknown')} days"
            star_type = classify_star(row.get("koi_steff", math.nan))

            planets.append({
                "name": str(name),
                "size": size,
                "period": period,
                "starType": star_type
            })

        return jsonify(planets)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ======== MAIN ========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

