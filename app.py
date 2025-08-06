import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Bone Fracture Detector",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .fractured {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .not-fractured {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model architecture
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        
        # Load trained weights
        model.load_state_dict(torch.load('bone_fracture_model.pth', map_location=device))
        model.eval()
        model.to(device)
        
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess the uploaded image"""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    return preprocess(image).unsqueeze(0)

def predict_fracture(model, image_tensor, device):
    """Make prediction on the image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()


def main():
    
    st.markdown('<h1 class="main-header">🦴 Bone Fracture Detection System</h1>', unsafe_allow_html=True)
    
    
    with st.sidebar:
        st.header("About This Project")
        st.write("""
        This AI-powered system can detect bone fractures in X-ray images using deep learning.
        
        **How it works:**
        1. Upload an X-ray image
        2. Our ResNet-18 model analyzes it
        3. Get instant results with confidence scores
        
        **Model Details:**
        - Architecture: ResNet-18
        - Training Dataset: Kaggle Bone Fracture Dataset
        - Accuracy: ~96.6% on test data
        """)
        
        st.header("Instructions")
        st.write("""
        1. Upload a clear X-ray image (PNG, JPG, JPEG)
        2. Wait for processing
        3. View the prediction results
        
        **Note:** This is for educational purposes only. Always consult medical professionals for actual diagnosis.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload X-ray Image")
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear X-ray image for fracture detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
            
            
            st.write("")
            
            # Process button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Load model
                    model, device = load_model()
                    
                    if model is not None:
                        # Preprocess and predict
                        try:
                            processed_image = preprocess_image(image)
                            prediction, confidence, probabilities = predict_fracture(model, processed_image, device)
                            
                            # Store results in session state
                            st.session_state.prediction = prediction
                            st.session_state.confidence = confidence
                            st.session_state.probabilities = probabilities
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
    
    with col2:
        st.header("Prediction Results")
        
        if hasattr(st.session_state, 'prediction'):
            prediction = st.session_state.prediction
            confidence = st.session_state.confidence
            probabilities = st.session_state.probabilities
            
            # Results display
            class_names = ["Fractured", "Not Fractured"]
            predicted_class = class_names[prediction]
            
            # Prediction box with styling
            if prediction == 0:  # Fractured
                st.markdown(f"""
                <div class="prediction-box fractured">
                    <h2>FRACTURE DETECTED!!</h2>
                    <h3>Confidence: {confidence:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
            else:  # Not Fractured
                st.markdown(f"""
                <div class="prediction-box not-fractured">
                    <h2> No Fracture detected</h2>
                    <h3>Confidence: {confidence:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence breakdown
            st.subheader("Confidence Breakdown")
            
            for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                st.write(f"**{class_name}**: {prob:.1%}")
                st.progress(prob)
            
            # Recommendation
            st.subheader("Recommendation")
            if prediction == 0:
                st.warning(" Potential fracture detected. Please consult a medical professional immediately for proper diagnosis and treatment.")
            else:
                st.success(" No obvious fracture detected. However, if you're experiencing pain or discomfort, please consult a healthcare provider.")
                
        else:
            st.info(" Upload an X-ray image and click 'Analyze Image' to see results")
            
            # Sample results placeholder
            st.subheader("Sample Analysis")
            st.write("Upload an image to see detailed prediction results here, including:")
            st.write("- Fracture detection status")
            st.write("- Confidence scores")
            st.write("- Medical recommendations")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Built using PyTorch & Streamlit</p>
            <p><em>For educational purposes only - Not a substitute for professional medical advice</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()
