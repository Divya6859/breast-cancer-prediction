import gradio as gr
import pickle
import numpy as np

# Load model and scaler
log_reg = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Actual feature names from dataset
feature_names = [
 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
 'smoothness_mean', 'compactness_mean', 'concavity_mean',
 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
 'smoothness_se', 'compactness_se', 'concavity_se',
 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
 'smoothness_worst', 'compactness_worst', 'concavity_worst',
 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Prediction function
def predict_tumor(*values):
    # Convert input to array
    values = np.array(values).reshape(1, -1)

    # Scale values
    values_scaled = scaler.transform(values)

    # Predict
    prediction = log_reg.predict(values_scaled)[0]
    if prediction == 1:
        return "🚨 Malignant (Cancerous)"
    else:
        return "✅ Benign (Non-Cancerous)"

# Create Gradio input fields using real column names
inputs = [gr.Number(label=name) for name in feature_names]

# Build Gradio UI
app = gr.Interface(
    fn=predict_tumor,
    inputs=inputs,
    outputs="text",
    title="Breast Cancer Prediction System",
    description="Enter the tumor feature values to predict whether it is Malignant or Benign."
)

app.launch()
