# AI Exo Hunter

**AI Exo Hunter** is a machine learning pipeline developed during the **NASA Space Apps Hackathon** to detect and classify **exoplanet transit events** from stellar light curve data.  
It combines data preprocessing, feature engineering, and predictive modeling using **LightGBM** and **CNN** architectures, served through a lightweight **Flask API** for real-time inference.

---

##  Project Overview

AI Exo Hunter aims to automate the detection and validation of **exoplanet candidates** by analyzing key astrophysical parameters derived from telescope observations.  
Our system processes raw astronomical data, extracts relevant features (such as orbital period, transit depth, stellar temperature, and luminosity), and predicts the likelihood of a real exoplanet.

---

## Motivation

Exoplanet detection traditionally requires extensive manual validation of transit signals in light curves.  
AI Exo Hunter reduces this bottleneck by providing a **data-driven classification** approach capable of scaling to thousands of potential signals.  
This allows scientists to **prioritize candidates** for observational follow-up, helping accelerate exoplanet discovery.

---

##  Core Components

| Component | Description |
|------------|-------------|
| **Preprocessing** | Cleans and normalizes raw light curve data, removing outliers and artifacts. |
| **Feature Extraction** | Computes astrophysical metrics (e.g., `temperature`, `stellar_gravity`, `planet_radius`, `period`). |
| **LightGBM Model** | Classifies each observation as *exoplanet* or *false positive* with calibrated probabilities. |
| **Flask API** | Provides endpoints for prediction and exploration of confirmed planets. |
| **Blockchain Integration (Optional)** | For research integrity — model and dataset hashes can be stored for verification. |


---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/AI-Exo-Hunter.git
cd AI-Exo-Hunter
```

### 2. Create a virtual environment
```
python -m venv .venv
source .venv/bin/activate  # For Linux/macOS
.venv\Scripts\activate     # For Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Run the Flask API
```
python API_flask_server.py
```

## API Endpoints

### GET/ 
```
curl http://localhost:7860/
```

### POST /predict
``` json
{
  "temperature": 493,
  "stellar_temp": 282,
  "depth": 3332,
  "stellar_gravity": 45984,
  "duration": 343,
  "stellar_radius": 3472,
  "magnitude": 7473234,
  "insol": 283278274,
  "period": 34352,
  "planet_radius": 1.2
}
```
### Response Example
```
{
  "prediction": 1,
  "probability": 0.9871
}
```
### GET /confirmed
## Retrieve a list of 20 confirmed exoplanets from NASA’s Kepler dataset.
```
[
  {
    "name": "Kepler-22b",
    "size": "Super-Earth",
    "period": "289.86 days",
    "starType": "G-type Star (Sun-like)"
  },
  ...
]
```
## 🔬 Model Description

### **Model 1: LightGBM**
- **Input:** tabular astrophysical features  
- **Output:** binary classification (`0` = false positive, `1` = exoplanet)  
- **Features used:**  
 temperature, stellar_temp, depth, stellar_gravity, duration,
 stellar_radius, magnitude, insol, period, planet_radius
- **Training data:** Real NASA data from Kepler and TESS missions (confirmed and false-positive samples)

---

### **Model 2: CNN (Prototype)**
- Processes **phase-folded light curves** as time-series inputs  
- Captures **temporal transit patterns** missed by tabular models  
- Integrated in `model.py` and `model.pt` (future release)

---

## 🧩 Example Workflow

1. Collect raw CSV or FITS light curve data  
2. Use preprocessing scripts to clean and normalize  
3. Extract relevant numerical features  
4. Pass the JSON data to the `/predict` endpoint  
5. Retrieve prediction probability and classification result  
6. Cross-check with the `/confirmed` endpoint to identify known planets  

---

## 📊 Evaluation Metrics

| **Metric** | **Description** |
|-------------|-----------------|
| **Accuracy** | Correct predictions over total samples |
| **Precision / Recall / F1** | For imbalanced datasets |
| **ROC AUC** | Discrimination ability between exoplanet and false positive |
| **Feature Importance** | Top contributing parameters for prediction |

---

## 🪙 Blockchain Integration (Optional)

To ensure **traceability** and **scientific reproducibility**, model artifacts and datasets can be hashed and stored in a **blockchain ledger**.  
This guarantees:
- Model integrity (no tampering of trained weights)  
- Dataset version verification  
- Transparent audit trail for every experiment  

---

## 🧾 Example Input Dataset

| **Feature** | **Description** |
|--------------|-----------------|
| `temperature` | Estimated planetary temperature (K) |
| `stellar_temp` | Host star temperature (K) |
| `depth` | Transit depth (ppm) |
| `stellar_gravity` | Surface gravity of the star (m/s²) |
| `duration` | Transit duration (hours) |
| `stellar_radius` | Radius of the host star (solar radii) |
| `magnitude` | Stellar apparent magnitude |
| `insol` | Insolation flux |
| `period` | Orbital period (days) |
| `planet_radius` | Radius of the planet (Earth radii) |

---

## 🧪 Future Work

- Integrate CNN predictions for hybrid ensemble scoring  
- Expand dataset with additional Kepler, K2, and TESS missions  
- Add real-time visualization dashboard  
- Deploy to cloud (AWS / Azure) with GPU inference  
- Blockchain integration for model audit trails  


---

## 🙌 Acknowledgments

- **NASA Exoplanet Archive** — for providing the primary dataset  
- **NASA Space Apps Challenge Team** — for organizing the hackathon  
- Open-source contributors of **LightGBM**, **Flask**, and **scikit-learn**  

---

### 💫 “Exploring distant worlds, one data point at a time.”







