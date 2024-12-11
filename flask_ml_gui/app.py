import pickle
from flask import Flask, render_template, request
import numpy as np
from logistic_regression import log_reg  # Import the trained model directly

app = Flask(__name__)

with open("log_reg_model.pkl", "rb") as file:
    log_reg = pickle.load(file)

# Define metadata for features with dropdown options
metadata = [
    {
        "name": "VisitorType",
        "description": "Visitor Type",
        "type": "dropdown",
        "options": ["New Visitor", "Returning Visitor", "Other"]
    },
    {
        "name": "Quarter",
        "description": "Quarter",
        "type": "dropdown", 
        "options": ["Q1", "Q2", "Q3", "Q4"]
    },
    {
        "name": "ProductRelated_Duration",
        "description": "Duration of product-related browsing",
        "type": "number"
    },
    {
        "name": "BounceRates",
        "description": "Bounce rate as a fraction",
        "type": "number"
    },
    {
        "name": "ExitRates",
        "description": "Exit rate as a fraction",
        "type": "number"
    },
    {
        "name": "TrafficType",
        "description": "Traffic type",
        "type": "number"
    }
]

@app.route("/")
def index():
    return render_template("index.html", metadata=metadata)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Prepare input features
        input_features = []

        # Handle Visitor Type encoding
        visitor_type = request.form.get("VisitorType")
        input_features.extend([
            1 if visitor_type == "New Visitor" else 0,
            1 if visitor_type == "Other" else 0,
            1 if visitor_type == "Returning Visitor" else 0
        ])

        # Handle Quarter encoding
        quarter = request.form.get("Quarter")
        input_features.extend([
            1 if quarter == "Q1" else 0,
            1 if quarter == "Q2" else 0,
            1 if quarter == "Q3" else 0,
            1 if quarter == "Q4" else 0
        ])

        # Add remaining numerical features
        remaining_features = [
            "ProductRelated_Duration", 
            "BounceRates", 
            "ExitRates", 
            "TrafficType"
        ]
        
        for feature in remaining_features:
            value = float(request.form.get(feature, 0))
            input_features.append(value)

        # Convert to numpy array
        input_features = np.array(input_features).reshape(1, -1)

        # Predict using the imported logistic regression model
        predicted_class = log_reg.predict(input_features)[0]
        probabilities = log_reg.predict_proba(input_features)[0]

        prediction_label = "Likely to Buy Something" if predicted_class == 1 else "Unlikely to Buy Something"

        return render_template(
            "results.html",
            prediction=prediction_label,
            prob_class_0=f"{probabilities[0] * 100:.2f}%",
            prob_class_1=f"{probabilities[1] * 100:.2f}%",
        )
    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True)