import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

from download_model import download_model

# —————————————————————————————
st.title("🍎 Fruit Classification App")

# Step 1: Download the model
download_model()

MODEL_PATH = "fruit_vgg16_model.keras"

# Step 2: Validate model file
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please check Google Drive link and download logic.")
    st.stop()

file_size = os.path.getsize(MODEL_PATH)
st.write(f"Model file size (bytes): {file_size}")

if file_size < 100_000_000:
    st.error("❌ Model file seems corrupted or incomplete (too small).")
    st.stop()

st.success("✅ Model file is present and valid. Loading model...")

# Step 3: Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("❌ Failed to load model. The model file may be corrupted or invalid.")
    st.code(str(e))
    st.stop()

# Step 4: Load class names
def load_class_names(file_path="class_names.txt"):
    if not os.path.exists(file_path):
        st.error("❌ class_names.txt not found.")
        st.stop()
    with open(file_path, "r") as f:
        return f.read().splitlines()

class_names = load_class_names()

# Step 5: Upload image and predict
uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    top_index = predictions.argmax()
    predicted_class = class_names[top_index]
    confidence = predictions[top_index] * 100

    st.success(f"Prediction: **{predicted_class}** ({confidence:.2f}%)")
