# =========================================================
# app.py
# Garbage Classification for Smart Cities
# =========================================================

import streamlit as st
import numpy as np
import cv2
import base64
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="wide"
)

# =========================================================
# LOAD YOUR OWN BACKGROUND IMAGE
# =========================================================
def get_base64(image_file):

    with open(image_file, "rb") as file:
        data = file.read()

    return base64.b64encode(data).decode()

# =========================================================
# IMPORTANT:
# Put your image in project folder
# Rename image as:
# background.jpg
# =========================================================
background_image = get_base64("BACK.jpg")

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(f"""
<style>

/* =====================================================
BACKGROUND IMAGE
===================================================== */

[data-testid="stAppViewContainer"]{{
    
    background-image:
    linear-gradient(
        rgba(0,0,0,0.55),
        rgba(0,0,0,0.55)
    ),
    url("data:image/jpg;base64,{background_image}");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* =====================================================
REMOVE HEADER BACKGROUND
===================================================== */

[data-testid="stHeader"]{{
    background: rgba(0,0,0,0);
}}

/* =====================================================
TITLE
===================================================== */

.main-title{{
    text-align:center;
    font-size:65px;
    font-weight:900;
    color:white;
    margin-top:20px;
    text-shadow:3px 3px 20px black;
    letter-spacing:1px;
}}

/* =====================================================
SUB TITLE
===================================================== */

.sub-title{{
    text-align:center;
    font-size:24px;
    color:#d7ffd9;
    margin-bottom:40px;
    line-height:1.8;
    text-shadow:2px 2px 10px black;
}}

/* =====================================================
GLASS CARD
===================================================== */

.glass-box{{
    background: rgba(255,255,255,0.10);
    border:1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(10px);
    padding:25px;
    border-radius:22px;
    color:white;
    box-shadow:0px 8px 32px rgba(0,0,0,0.4);
}}

/* =====================================================
UPLOAD TEXT
===================================================== */

.upload-text{{
    color:white;
    font-size:26px;
    font-weight:bold;
}}

/* =====================================================
BUTTON
===================================================== */

.stButton>button{{
    width:100%;
    background: linear-gradient(90deg,#00c853,#64dd17);
    color:white;
    border:none;
    padding:14px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
    transition:0.3s;
}}

.stButton>button:hover{{
    background: linear-gradient(90deg,#00e676,#76ff03);
    color:black;
    transform:scale(1.02);
}}

/* =====================================================
RESULT BOX
===================================================== */

.result-box{{
    background: rgba(0,0,0,0.60);
    padding:25px;
    border-radius:20px;
    color:white;
    margin-top:20px;
    text-align:center;
}}

/* =====================================================
FOOTER
===================================================== */

.footer{{
    text-align:center;
    color:white;
    font-size:18px;
    margin-top:50px;
    padding:20px;
}}

/* =====================================================
IMAGE STYLE
===================================================== */

img{{
    border-radius:15px;
}}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model_file():

    model = load_model("garbage_classifier_model.h5")

    return model

model = load_model_file()

# =========================================================
# CLASS NAMES
# =========================================================
class_names = [
    'cardboard',
    'glass',
    'metal',
    'paper',
    'plastic',
    'trash'
]

biodegradable = ['paper', 'cardboard']

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_image(image):

    img = np.array(image)

    img_resized = cv2.resize(img, (224, 224))

    img_array = preprocess_input(img_resized)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction) * 100

    if predicted_class in biodegradable:
        waste_type = "Biodegradable"
    else:
        waste_type = "Non-Biodegradable"

    return predicted_class, waste_type, confidence

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="main-title">
♻️ Smart Waste Classification
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">

🌍 City of Green, Not City of Trash 🌱 <br>
AI Powered Waste Detection for Cleaner & Sustainable Smart Cities

</div>
""", unsafe_allow_html=True)

# =========================================================
# MAIN SECTION
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([1,1])

# =========================================================
# LEFT SIDE
# =========================================================
with col1:

    st.markdown("""
    <div class="glass-box">
    <div class="upload-text">
    📤 Upload Garbage Image
    </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"]
    )

# =========================================================
# RIGHT SIDE
# =========================================================
with col2:

    st.markdown("""
    <div class="glass-box">

    <h2>🌱 Smart Waste Management</h2>

    <p style="font-size:18px; line-height:1.9;">

    ✔ AI Based Waste Detection <br>
    ✔ Automatic Waste Classification <br>
    ✔ Smart Recycling Support <br>
    ✔ Reduces Environmental Pollution <br>
    ✔ Creates Green Smart Cities

    </p>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PREDICTION SECTION
# =========================================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<br>", unsafe_allow_html=True)

    img_col, pred_col = st.columns(2)

    # =====================================================
    # IMAGE DISPLAY
    # =====================================================
    with img_col:

        st.markdown("""
        <div class="glass-box">
        <h2 style="color:white;">🖼 Uploaded Image</h2>
        </div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            width=320
        )

    # =====================================================
    # PREDICTION DISPLAY
    # =====================================================
    with pred_col:

        st.markdown("""
        <div class="glass-box">
        <h2 style="color:white;">🔍 AI Prediction</h2>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Classify Waste"):

            with st.spinner("Analyzing Waste Image..."):

                predicted_class, waste_type, confidence = predict_image(image)

            st.markdown(f"""
            <div class="result-box">

            <h1>♻️ {predicted_class.upper()}</h1>

            <h2>Waste Type: {waste_type}</h2>

            <h2>Confidence: {confidence:.2f}%</h2>

            </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence))

            if waste_type == "Biodegradable":

                st.success("🌿 Eco-Friendly Waste Detected!")

            else:

                st.error("⚠ Non-Biodegradable Waste Detected!")

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">

🌍 Garbage Classification for Smart Cities <br>
Powered by <b>Streamlit • TensorFlow • MobileNetV2</b>

</div>
""", unsafe_allow_html=True)