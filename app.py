from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(x) for x in request.form.values()]
        features = np.array([values])

        prediction = model.predict(features)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted PM Temperature: {round(prediction, 2)}"
        )

    except:
        return render_template("index.html", prediction_text="❌ Invalid Input")

if __name__ == "__main__":
    app.run(debug=True)