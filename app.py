import os
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

from models import create_model, AVAILABLE_MODELS

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Visiontte",
    layout="centered"
)

# ===============================
# Load CSS
# ===============================
def load_css(path):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(BASE_DIR, "Interface", "static", "style.css")
load_css(CSS_PATH)

# ===============================
# Header
# ===============================
st.markdown("<div class='container'>", unsafe_allow_html=True)

st.markdown("<h1>Visiontte</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>AI-powered system for detecting manipulated low-light facial images.</p>",
    unsafe_allow_html=True
)

# ===============================
# Model selection
# ===============================
model_name = st.selectbox(
    "اختر الموديل",
    AVAILABLE_MODELS
)

# ===============================
# Load model (cached)
# ===============================
@st.cache_resource
def load_cached_model(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(name).to(device)

    weight_map = {
        "resnet34_se_only": "resnet34_se_only.pth",
        "lideepdet": "lideepdet.pth",
        "efficientnetb0": "EfficientNetB0.pth",
        "xception": "xception.pth",
    }

    weight_path = os.path.join(BASE_DIR, "Models", weight_map[name])

    checkpoint = torch.load(
    weight_path,
    map_location=device,
    weights_only=False
)


    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, device


model, device = load_cached_model(model_name)

# ===============================
# Transform
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===============================
# Upload image
# ===============================
uploaded_file = st.file_uploader(
    "📤 ارفع صورة",
    type=["jpg", "png", "jpeg"]
)

analyze = st.button("Analyze Image")

# ===============================
# Inference
# ===============================
if uploaded_file is not None and analyze:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة المدخلة", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        pred = torch.argmax(logits, dim=1).item()

    if pred == 1:
        st.markdown(
            f"<div class='result fake'>🚨 Fake (Probability: {prob:.3f})</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result real'>✅ Real (Fake Probability: {prob:.3f})</div>",
            unsafe_allow_html=True
        )

    st.progress(prob)

# ===============================
# Footer
# ===============================
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<footer>Developed for academic research purposes</footer>",
    unsafe_allow_html=True
)
