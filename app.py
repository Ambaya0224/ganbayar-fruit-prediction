import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from download_model import download_model  # make sure this exists

# ----------------------------
# APP TITLE
# ----------------------------
st.title("🍎 Fruit Classification App")

# ----------------------------
# DOWNLOAD MODEL
# ----------------------------
download_model()

MODEL_PATH = "fruit_vgg16_model.keras"

# Validate model file
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Download may have failed.")
    st.stop()

if os.path.getsize(MODEL_PATH) < 100_000_000:
    st.error("❌ Model file appears corrupted or incomplete.")
    st.stop()

st.success("Model file downloaded and verified.")

# ----------------------------
# LOAD MODEL
# ----------------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model loaded successfully! 🎉")
except Exception as e:
    st.error("❌ Failed to load model. The file may be corrupted.")
    st.code(str(e))
    st.stop()

# ----------------------------
# LOAD CLASS NAMES
# ----------------------------
def load_class_names(file_path="class_names.txt"):
    try:
        with open(file_path, "r") as f:
            return f.read().splitlines()
    except Exception as e:
        st.error("❌ Could not load class_names.txt.")
        st.code(str(e))
        st.stop()

class_names = load_class_names()

# ----------------------------
# IMAGE UPLOADER & PREDICTION
# ----------------------------
uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize to match model input
    target_size = (64, 64)  # match your model input
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    top_index = predictions.argmax()
    predicted_class = class_names[top_index]
    confidence = predictions[top_index] * 100

    st.success(f"Prediction: **{predicted_class}** ({confidence:.2f}%)")
