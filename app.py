import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load trained model
model = tf.keras.models.load_model('brain_tumor_model.h5')

# Streamlit app
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image to classify if it shows a Brain Tumor or not.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    img_path = os.path.join("uploaded_image.jpg")
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display image
    st.image(img_path, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    label = "🧠 Tumor Detected" if prediction > 0.5 else "✅ No Tumor Detected"

    # Show result
    st.subheader("Result:")
    st.success(label)

    

   
