import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from download_model import download_model  # <-- import here

st.title("🍎 Fruit Classification App")

# Step 1: Download the model if not exists
download_model()

# Step 2: Load the model locally
model = tf.keras.models.load_model("fruit_vgg16_model.keras")

# Step 3: Load class names
def load_class_names(file_path="class_names.txt"):
    with open(file_path, "r") as f:
        return f.read().splitlines()

class_names = load_class_names()

# Step 4: Upload image and predict
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

    st.success(f"Prediction: {predicted_class} ({confidence:.2f}%)")
