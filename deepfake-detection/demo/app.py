import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Deepfake Face Detector", layout="centered")

st.title("Deepfake Face Detector")
st.write("Upload any photo containing a face — the model will automatically detect, crop, and classify it.")

with st.expander("What is a deepfake?"):
    st.write(
        "A **deepfake** is a manipulated image or video where a face has been digitally "
        "altered — usually by swapping one person's face onto another person's body/video "
        "(face-swap), or by altering someone's expressions and mouth movements to make it "
        "look like they said or did something they didn't (reenactment). This tool checks "
        "specifically for **face-swap style manipulation**."
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_v2_moredata.pt")
THRESHOLD = 0.45

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, 1))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_face_detector():
    return MTCNN(keep_all=False, device=DEVICE, margin=40)

model = load_model()
mtcnn = load_face_detector()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def confidence_zone(pct):
    """Map a confidence percentage to a plain-language reliability label."""
    if pct >= 85:
        return "High confidence"
    elif pct >= 65:
        return "Moderate confidence"
    else:
        return "Low confidence — borderline result"

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert("RGB")
    st.image(raw_image, caption="Uploaded Image", width='stretch')

    boxes, probs = mtcnn.detect(np.array(raw_image))

    if boxes is None or len(boxes) == 0:
        st.error("No face detected in this image. Please upload a clearer photo with a visible face.")
    else:
        best_idx = probs.argmax()
        if probs[best_idx] < 0.90:
            st.warning(f"Face detected with low confidence ({probs[best_idx]:.2f}). Result below may be unreliable.")

        x1, y1, x2, y2 = [int(v) for v in boxes[best_idx]]
        w, h = raw_image.size
        m = 40
        x1, y1 = max(0, x1 - m), max(0, y1 - m)
        x2, y2 = min(w, x2 + m), min(h, y2 + m)
        face_crop = raw_image.crop((x1, y1, x2, y2))

        st.image(face_crop, caption="Detected Face (what the model actually sees)", width=200)

        input_tensor = transform(face_crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor).squeeze(1)
            prob = torch.sigmoid(output).item()

        is_fake = prob >= THRESHOLD
        label = "FAKE" if is_fake else "REAL"
        confidence_pct = (prob if is_fake else (1 - prob)) * 100
        zone = confidence_zone(confidence_pct)

        st.markdown("---")

        # --- Color-coded verdict ---
        if is_fake:
            st.markdown(
                f"<h2 style='color:#e63946;'>⚠️ Prediction: FAKE</h2>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h2 style='color:#2a9d8f;'>✅ Prediction: REAL</h2>",
                unsafe_allow_html=True
            )

        st.markdown(f"**{zone}** — {confidence_pct:.1f}% confidence")

        # --- Explanation of what the percentage means ---
        st.caption(
            f"This means the model is {confidence_pct:.1f}% certain this face is "
            f"{'a face-swap manipulation' if is_fake else 'authentic/unaltered'}, "
            f"based on patterns it learned from thousands of real and manipulated training images. "
            f"It is not a measure of image quality or how 'human-looking' the face is."
        )

        # --- Labeled real<->fake bar ---
        st.write("")
        col1, col2, col3 = st.columns([1, 8, 1])
        with col1:
            st.markdown("**Real**")
        with col2:
            st.progress(prob)
        with col3:
            st.markdown("**Fake**")
        st.caption(f"Raw model score: {prob:.3f} (0.0 = certainly real, 1.0 = certainly fake). "
                   f"Decision threshold: {THRESHOLD}")

        with st.expander("How this works & limitations"):
            st.write(
                "This model was trained on the FaceForensics++ Deepfakes subset using transfer "
                "learning (EfficientNet-B0). It reliably detects face-swap style manipulations "
                "but has reduced accuracy on other forgery types (e.g. expression-reenactment "
                "methods like Face2Face) it wasn't trained on — a known limitation documented "
                "in the project's generalization testing (accuracy drops from ~74% to ~45% on "
                "unseen manipulation types)."
            )
