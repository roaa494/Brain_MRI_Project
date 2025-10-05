import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image
import os

# Load model
MODEL_PATH = "C:/Users/iTech/Downloads/MRI/saved_model/brain_tumor_model.h5"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
else:
    st.error("⚠️ Model not found. Please run train_model.py first.")
    st.stop()

st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image to check for Brain Tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess image
    img = img.resize((150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        st.error("🚨 Tumor detected!")
    else:
        st.success("✅ No tumor detected.")
