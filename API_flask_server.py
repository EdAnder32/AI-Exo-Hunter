## An python API for our Model
## Design by edander32 (Edmilson Alexandre) and jjambo(Joaquim Jambo)

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("exoplanet_model.pkl")

@app.route("/")
def home():
    return {"message": "API do classificador de exoplanetas online."}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]
        return jsonify({"prediction": int(pred), "probability": float(prob)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

