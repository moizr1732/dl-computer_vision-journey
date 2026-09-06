import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

st.title("Deepfake Face Detector")
st.write("Upload any photo containing a face — the model will automatically detect, crop, and classify it.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.45  # calibrated from validation tuning, not the default 0.5

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, 1))
    model.load_state_dict(torch.load("best_model_v2_moredata.pt", map_location=DEVICE))
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

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert("RGB")
    st.image(raw_image, caption="Uploaded Image", use_container_width=True)

    boxes, probs = mtcnn.detect(np.array(raw_image))

    if boxes is None or len(boxes) == 0:
        st.error("No face detected in this image. Please upload a clearer photo with a visible face.")
    else:
        best_idx = probs.argmax()
        if probs[best_idx] < 0.90:
            st.warning(f"Face detected with low confidence ({probs[best_idx]:.2f}). Result may be unreliable.")

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

        label = "FAKE" if prob >= THRESHOLD else "REAL"
        confidence = prob if label == "FAKE" else (1 - prob)

        st.write(f"### Prediction: {label}")
        st.write(f"Confidence: {confidence:.1%}")
        st.progress(prob)

        with st.expander("How this works"):
            st.write(
                "This model was trained on the FaceForensics++ Deepfakes subset using transfer "
                "learning (EfficientNet-B0). It reliably detects face-swap style manipulations "
                "but has reduced accuracy on other forgery types (e.g. expression-reenactment "
                "methods like Face2Face) it wasn't trained on — a known limitation documented "
                "in the project's generalization testing."
            )
