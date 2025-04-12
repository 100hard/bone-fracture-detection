import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


device = "cpu"
@st.cache_resource  
def load_model():
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("bone_fracture_model.pth", map_location=device))
    model.eval()
    return model

model = load_model()


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


st.title("Bone Fracture Detection Dashboard")
st.markdown("""
Upload an X-Ray image to detect fractures using a ResNet18 model trained with 96% accuracy.
""")


uploaded_file = st.file_uploader("Choose an X-Ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
  
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Uploaded X-Ray", use_column_width=True)

   
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = "Fractured" if predicted.item() == 0 else "Not Fractured"
    
    
    st.subheader("Prediction")
    st.write(f"Result: **{prediction}**", unsafe_allow_html=True)

    
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Test Accuracy:** 96%")
    with col2:
        st.write("**Dataset:** 10k+ X-Ray images")