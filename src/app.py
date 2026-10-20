import json
from pathlib import Path
from PIL import Image
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

# Set page config
st.set_page_config(
    page_title="Bird Species Classifier",
    page_icon="🦜",
    layout="centered"
)

st.title("🦜 CUB-200 Bird Species Classifier")
st.write("Fine-grained classification across 200 bird species using **Swin-T**.")

# 1. Load Class Labels
@st.cache_data
def load_class_names():
    with open("classes.json", "r") as f:
        return json.load(f)

class_names = load_class_names()

# 2. Load the Trained Model
@st.cache_resource
def load_classifier(ckpt_path: str):
    # Swin-T architecture matching training
    model = models.swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, 200)
    
    # Load weights onto CPU for broad deployment compatibility
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    # Support full checkpoint dictionary or raw weights
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 3. Image Preprocessing Transform
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Sidebar Info
st.sidebar.header("Model Performance")
st.sidebar.markdown("""
- **Architecture:** Swin-T (Hierarchical ViT)
- **Top-1 Test Accuracy:** **86.05%**
- **Avg Class Accuracy:** **85.78%**
- **Top-5 Accuracy:** **96.10%**
""")

ckpt_file = "swin_t_best.pth"
if not Path(ckpt_file).exists():
    st.error(f"Checkpoint '{ckpt_file}' not found in the current directory.")
    st.stop()

model = load_classifier(ckpt_file)

# Image Uploader
uploaded_file = st.file_uploader("Upload an image of a bird...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Inference
    tensor = eval_transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    
    top5_probs, top5_indices = torch.topk(probs, k=5)
    
    # Format Top Prediction
    best_species = class_names[top5_indices[0].item()]
    best_prob = top5_probs[0].item() * 100
    
    st.success(f"**Top Prediction:** {best_species} ({best_prob:.1f}% confidence)")
    
    # Top-5 Chart
    df_results = pd.DataFrame({
        "Species": [class_names[idx.item()] for idx in top5_indices],
        "Confidence (%)": [p.item() * 100 for p in top5_probs]
    }).sort_values(by="Confidence (%)", ascending=True)

    st.subheader("Top-5 Predictions")
    st.bar_chart(data=df_results.set_index("Species"), horizontal=True)