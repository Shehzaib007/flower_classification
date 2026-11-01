import os
import sys

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
MODEL_PATH = "flowers_cnn.h5"          # change if your file has another name
IMG_SIZE = (128, 128)                  # must match training size
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]  # training order
LOW_CONFIDENCE = 0.60

FLOWERS = {
    "daisy":     {"emoji": "🌼", "color": "#F5B700",
                  "fact": "Daisy flowers close at night and open again with the morning sun. The name comes from 'day's eye'."},
    "dandelion": {"emoji": "🌿", "color": "#F28C28",
                  "fact": "Every part of a dandelion is edible, and its seed head can carry seeds on the wind for miles."},
    "rose":      {"emoji": "🌹", "color": "#E63462",
                  "fact": "Roses have been grown for over 5,000 years, and there are more than 300 species."},
    "sunflower": {"emoji": "🌻", "color": "#F9A620",
                  "fact": "Young sunflowers follow the sun across the sky, a movement called heliotropism."},
    "tulip":     {"emoji": "🌷", "color": "#C2185B",
                  "fact": "In 17th-century Holland, tulip bulbs were so valuable they were traded like gold."},
}

# Project links (edit the labels here if you like)
LINKS = [
    {"icon": "🌐", "label": "Live App",
     "desc": "Open the deployed web app",
     "url": "https://flowerclassification01.streamlit.app/"},
    {"icon": "📁", "label": "Project File",
     "desc": "View the file on Google Drive",
     "url": "https://drive.google.com/file/d/1RV6hphcY9I6nCKSaIqSsgQRGLoqb5Mz2/view?usp=drive_link"},
    {"icon": "📊", "label": "Presentation",
     "desc": "View the project slides",
     "url": "https://docs.google.com/presentation/d/1VDhBnrQzXjfvuXAyWYOQC2R5-ZNw_OGl/edit?usp=drive_link&ouid=104317572637074554973&rtpof=true&sd=true"},
]

st.set_page_config(page_title="Flower Recognition AI", page_icon="🌸", layout="wide")

# ----------------------------------------------------------------
# Styling
# ----------------------------------------------------------------
CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.stApp {background: linear-gradient(160deg, #fff5f7 0%, #fdf2ff 45%, #f0f9ff 100%);}
.block-container {padding-top: 2rem; max-width: 1100px;}

.hero {
    background: linear-gradient(135deg, #ff6b9d 0%, #ff8e72 55%, #ffc371 100%);
    border-radius: 28px; padding: 2.6rem 1.5rem; text-align: center;
    color: #ffffff; box-shadow: 0 18px 40px rgba(255, 107, 157, 0.30);
    margin-bottom: 1.8rem;
}
.hero h1 {color: #ffffff; font-size: 2.8rem; margin: 0 0 .4rem 0; font-weight: 800;}
.hero p  {color: #fff4f0; font-size: 1.1rem; margin: 0;}

.card {
    background: #ffffff; border-radius: 22px; padding: 1.5rem;
    box-shadow: 0 8px 28px rgba(80, 60, 120, 0.10); color: #1f2937;
    margin-bottom: 1.2rem;
}
.result-emoji {font-size: 4.2rem; line-height: 1; margin-bottom: .2rem;}
.result-name  {font-size: 2.4rem; font-weight: 800; margin: 0; text-transform: capitalize;}
.badge {
    display: inline-block; padding: .35rem .95rem; border-radius: 999px;
    color: #ffffff; font-weight: 700; font-size: .95rem; margin: .6rem 0 1rem 0;
}
.fact {
    background: #faf5ff; border-left: 5px solid #a855f7; border-radius: 12px;
    padding: .8rem 1rem; color: #4b5563; font-size: .95rem;
}
.warn {
    background: #fff7ed; border-left: 5px solid #f97316; border-radius: 12px;
    padding: .8rem 1rem; color: #9a3412; font-size: .95rem; margin-top: .8rem;
}
.section-title {font-size: 1.2rem; font-weight: 700; margin: 0 0 1rem 0; color: #1f2937;}

.bar-row {display: flex; align-items: center; gap: .8rem; margin-bottom: .7rem;}
.bar-label {width: 120px; font-weight: 600; color: #374151; font-size: .95rem;}
.bar-track {flex: 1; background: #eef0f4; border-radius: 999px; height: 14px; overflow: hidden;}
.bar-fill  {height: 100%; border-radius: 999px; animation: grow .9s ease-out;}
.bar-pct   {width: 58px; text-align: right; font-weight: 700; color: #374151; font-size: .9rem;}
@keyframes grow {from {width: 0;}}

.chip {
    display: inline-block; background: #ffffff; border-radius: 999px;
    padding: .3rem .8rem; margin: .2rem .15rem; font-size: .9rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.08); color: #374151;
}
.empty {text-align: center; padding: 2.5rem 1rem; color: #6b7280;}
.empty .big {font-size: 3.5rem;}
.footer {text-align: center; color: #9ca3af; font-size: .85rem; margin-top: 2rem;}

.link-card {
    display: block; background: #ffffff; border-radius: 18px; padding: 1.1rem 1.2rem;
    text-decoration: none !important; box-shadow: 0 6px 20px rgba(80, 60, 120, 0.10);
    border: 2px solid transparent; transition: all .2s ease; height: 100%;
}
.link-card:hover {
    transform: translateY(-4px); border-color: #ff6b9d;
    box-shadow: 0 12px 28px rgba(255, 107, 157, 0.25);
}
.link-icon  {font-size: 2rem; line-height: 1;}
.link-title {font-weight: 700; color: #1f2937; font-size: 1.05rem; margin-top: .4rem;}
.link-desc  {color: #6b7280; font-size: .88rem;}
.side-link {
    display: block; background: #ffffff; border-radius: 12px; padding: .55rem .9rem;
    margin-bottom: .5rem; text-decoration: none !important; color: #374151 !important;
    font-weight: 600; box-shadow: 0 2px 8px rgba(0,0,0,.08); transition: all .2s ease;
}
.side-link:hover {background: #ff6b9d; color: #ffffff !important;}

div[data-testid="stImage"] img {border-radius: 18px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌸 About")
    st.write("A convolutional neural network (CNN) trained to recognise "
             "five common flower types from a photo.")
    st.markdown("**Supported flowers**")
    st.markdown("".join(
        f"<span class='chip'>{FLOWERS[c]['emoji']} {c.capitalize()}</span>"
        for c in CLASS_NAMES), unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**📸 Tips for best results**")
    st.markdown(
        "- Use a clear, well-lit photo\n"
        "- Keep one flower in the centre\n"
        "- Avoid heavy filters or blur"
    )
    st.markdown("---")
    st.markdown("**🔗 Project links**")
    st.markdown("".join(
        f"<a class='side-link' href='{l['url']}' target='_blank'>{l['icon']} {l['label']}</a>"
        for l in LINKS), unsafe_allow_html=True)
    st.markdown("---")
    with st.expander("Debug info"):
        st.write("Python:", sys.version.split()[0])
        st.write("Files:", sorted(os.listdir(".")))
        if os.path.exists(MODEL_PATH):
            st.write(f"Model size: {os.path.getsize(MODEL_PATH) / 1e6:.1f} MB")

# ----------------------------------------------------------------
# Hero
# ----------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🌸 Flower Recognition AI</h1>
        <p>Upload a photo and let deep learning tell you which flower it is.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------
# Model loading (errors are shown on the page, not as a crash)
# ----------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file `{MODEL_PATH}` not found in the repo root.")
    st.stop()

try:
    import tensorflow as tf
except Exception as e:
    st.error("TensorFlow failed to import:")
    st.exception(e)
    st.stop()


@st.cache_resource(show_spinner="Loading the flower model...")
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("The model file was found but could not be loaded:")
    st.exception(e)
    st.stop()


def preprocess(img):
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0   # same scaling as training
    return np.expand_dims(arr, axis=0)


def prob_bars(probs):
    order = np.argsort(probs)[::-1]
    rows = ""
    for i in order:
        name = CLASS_NAMES[i]
        info = FLOWERS[name]
        pct = float(probs[i]) * 100
        rows += (
            f"<div class='bar-row'>"
            f"<div class='bar-label'>{info['emoji']} {name.capitalize()}</div>"
            f"<div class='bar-track'><div class='bar-fill' "
            f"style='width:{pct:.1f}%; background:{info['color']}'></div></div>"
            f"<div class='bar-pct'>{pct:.1f}%</div></div>"
        )
    return rows


# ----------------------------------------------------------------
# Input
# ----------------------------------------------------------------
tab_upload, tab_camera = st.tabs(["📤  Upload a photo", "📷  Use camera"])
with tab_upload:
    uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png"],
                                label_visibility="collapsed")
with tab_camera:
    captured = st.camera_input("Take a picture", label_visibility="collapsed")

file = uploaded or captured

# ----------------------------------------------------------------
# Output
# ----------------------------------------------------------------
if file is None:
    st.markdown(
        """
        <div class="card empty">
            <div class="big">🌼 🌹 🌻 🌷</div>
            <h3 style="color:#374151;">No image yet</h3>
            <p>Upload or capture a flower photo above to see the prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    image = ImageOps.exif_transpose(Image.open(file))
    with st.spinner("Analysing your flower..."):
        probs = model.predict(preprocess(image), verbose=0)[0]

    idx = int(np.argmax(probs))
    name = CLASS_NAMES[idx]
    info = FLOWERS[name]
    conf = float(probs[idx])

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.image(image, caption="Your photo", use_container_width=True)

    with right:
        warning = (
            "<div class='warn'>⚠️ Low confidence. This may not be one of the "
            "five supported flowers, or the photo may be unclear.</div>"
            if conf < LOW_CONFIDENCE else ""
        )
        st.markdown(
            f"""
            <div class="card">
                <div class="result-emoji">{info['emoji']}</div>
                <p class="result-name" style="color:{info['color']}">{name}</p>
                <span class="badge" style="background:{info['color']}">
                    {conf * 100:.1f}% confident
                </span>
                <div class="fact">💡 <b>Did you know?</b> {info['fact']}</div>
                {warning}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="card">
            <p class="section-title">📊 Confidence breakdown</p>
            {prob_bars(probs)}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<p class='section-title' style='margin-top:2rem'>🔗 Project resources</p>",
            unsafe_allow_html=True)
link_cols = st.columns(len(LINKS), gap="medium")
for col, l in zip(link_cols, LINKS):
    with col:
        st.markdown(
            f"<a class='link-card' href='{l['url']}' target='_blank'>"
            f"<div class='link-icon'>{l['icon']}</div>"
            f"<div class='link-title'>{l['label']}</div>"
            f"<div class='link-desc'>{l['desc']}</div></a>",
            unsafe_allow_html=True,
        )

st.markdown("<div class='footer'>Built with TensorFlow &amp; Streamlit 🌱</div>",
            unsafe_allow_html=True)
