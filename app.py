import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
MODEL_PATH = "flowers_cnn.h5"  # change if your file is named flower_cnn.h5
IMG_SIZE = (128, 128)          # must match training size
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]  # same order as training

st.set_page_config(page_title="Flower Recognition", page_icon="🌸", layout="centered")


# ---------------------------------------------------------------
# Load model once (cached across reruns)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    return tf.keras.models.load_model(path, compile=False)


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # same rescaling as training
    return np.expand_dims(arr, axis=0)


def predict(model, img: Image.Image):
    preds = model.predict(preprocess(img), verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]), preds


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("🌸 Flower Recognition")
st.write("Upload a photo of a flower and the CNN will tell you what it is. "
         "Supported classes: " + ", ".join(c.capitalize() for c in CLASS_NAMES) + ".")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file `{MODEL_PATH}` not found. Place it in the same folder as `app.py`.")
    st.write("Files in this folder:", sorted(os.listdir(".")))
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("The model file was found but could not be loaded. Details below:")
    st.exception(e)
    st.caption(f"TensorFlow version running here: {tf.__version__}")
    st.stop()

source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
if source == "Upload":
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
else:
    file = st.camera_input("Take a picture")

if file is not None:
    image = Image.open(file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Your image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            label, conf, probs = predict(model, image)

        st.subheader(f"Prediction: {label.capitalize()}")
        st.metric("Confidence", f"{conf * 100:.1f}%")

        if conf < 0.60:
            st.warning("Low confidence. The image may not be one of the 5 supported flowers.")

    st.markdown("#### Class probabilities")
    df = pd.DataFrame({"Flower": [c.capitalize() for c in CLASS_NAMES], "Probability": probs})
    st.bar_chart(df.set_index("Flower"))
else:
    st.info("Upload an image to get started.")
