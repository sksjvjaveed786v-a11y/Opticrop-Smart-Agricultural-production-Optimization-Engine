from flask import Flask, render_template, request
import joblib
import pandas as pd
from crop_info import crop_details
from fertilizer_info import fertilizer_details

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

    prediction = model.predict(input_data)[0]

    confidence = max(model.predict_proba(input_data)[0]) * 100
    confidence = round(confidence, 2)
    info = crop_details.get(prediction.lower(), {})
    fertilizer = fertilizer_details.get(prediction.lower(), {})
    
    reasons = []

    if N > 80:
        reasons.append("✅ High Nitrogen level is suitable.")

    if P > 40:
        reasons.append("✅ Phosphorus level supports healthy crop growth.")

    if K > 40:
        reasons.append("✅ Potassium level is adequate.")

    if temperature >= 20 and temperature <= 30:
        reasons.append("✅ Temperature is in the ideal range.")

    if humidity >= 70:
        reasons.append("✅ Humidity is favorable.")

    if rainfall >= 150:
        reasons.append("✅ Rainfall is sufficient.")

    if ph >= 5.5 and ph <= 7.0:
        reasons.append("✅ Soil pH is suitable.")
    

    return render_template(
    "index.html",
    prediction=prediction,
    confidence=confidence,
    info=info,
    fertilizer=fertilizer,
    reasons=reasons
)

if __name__ == "__main__":
    app.run(debug=True)