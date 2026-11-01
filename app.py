import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

MODEL_PATH = "flowers_cnn.h5"  # change if your file is named flower_cnn.h5
IMG_SIZE = (128, 128)
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

st.set_page_config(page_title="Flower Recognition", page_icon="🌸", layout="centered")
st.title("🌸 Flower Recognition")

# ---- Diagnostics (remove once everything works) ----
with st.expander("Diagnostics"):
    st.write("Python:", sys.version)
    st.write("Files in app folder:", sorted(os.listdir(".")))
    if os.path.exists(MODEL_PATH):
        st.write(f"Model size: {os.path.getsize(MODEL_PATH) / 1e6:.1f} MB")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file `{MODEL_PATH}` not found in the repo root.")
    st.stop()

# ---- Import TensorFlow lazily so errors show on the page ----
try:
    import tensorflow as tf
    st.caption(f"TensorFlow {tf.__version__}")
except Exception as e:
    st.error("TensorFlow failed to import:")
    st.exception(e)
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("The model file was found but could not be loaded:")
    st.exception(e)
    st.stop()


def preprocess(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


st.write("Upload a photo of a flower. Classes: "
         + ", ".join(c.capitalize() for c in CLASS_NAMES) + ".")

source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
file = (st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if source == "Upload" else st.camera_input("Take a picture"))

if file is not None:
    image = Image.open(file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Your image", use_container_width=True)
    with col2:
        with st.spinner("Analyzing..."):
            probs = model.predict(preprocess(image), verbose=0)[0]
        idx = int(np.argmax(probs))
        st.subheader(f"Prediction: {CLASS_NAMES[idx].capitalize()}")
        st.metric("Confidence", f"{probs[idx] * 100:.1f}%")
        if probs[idx] < 0.60:
            st.warning("Low confidence. The image may not be a supported flower.")
    df = pd.DataFrame({"Flower": [c.capitalize() for c in CLASS_NAMES], "Probability": probs})
    st.bar_chart(df.set_index("Flower"))
else:
    st.info("Upload an image to get started.")
