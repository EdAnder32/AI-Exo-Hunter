from flask import Flask, request, jsonify
from flask_cors import cross_origin
import pandas as pd
import joblib
import torch

app = Flask(__name__)

# --- Modelo 1: Joblib (.pkl)
model_pkl = joblib.load("exoplanet_model.pkl")
NUM_FEATURES = model_pkl.n_features_in_
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

# --- Modelo 2: PyTorch (.pt) genérico
# Carrega o modelo completo salvo com torch.save(model)
model_pt = torch.load("model.pt", map_location=torch.device('cpu'))
model_pt.eval()

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    try:
        data = request.get_json()
        
        # --- Prepara os dados
        row = [0] * NUM_FEATURES
        for key, idx in IMPORTANT_FEATURES.items():
            if key in data:
                row[idx] = data[key]
        df = pd.DataFrame([row])
        
        # --- Previsão do modelo .pkl
        prob_pkl = model_pkl.predict_proba(df)[0][1]
        
        # --- Previsão do modelo .pt
        # Assumindo que o modelo retorna um tensor com probabilidade entre 0 e 1
        x_tensor = torch.tensor([row], dtype=torch.float32)
        with torch.no_grad():
            output = model_pt(x_tensor)
            # Caso o output seja logit, aplica sigmoid
            if output.shape[-1] == 1 or len(output.shape) == 2 and output.shape[1] == 1:
                prob_pt = torch.sigmoid(output).item()
            else:
                # Se for vetor de probabilidades (softmax), pega a classe 1
                prob_pt = torch.softmax(output, dim=1)[0][1].item()
        
        # --- Média das probabilidades
        prob_avg = (prob_pkl + prob_pt) / 2
        pred_final = int(prob_avg >= 0.5)
        
        return jsonify({
            "prediction": pred_final,
            "probability": prob_avg,
            "prob_pkl": prob_pkl,
            "prob_pt": prob_pt
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
