# Bone Fracture Detection System

An AI-powered web application that detects bone fractures in X-ray images using deep learning.

## Live Demo
[**Try the app here**](your-streamlit-app-url.streamlit.app)

## Project Overview

This project implements a computer vision classifier to detect bone fractures in X-ray images. Using a ResNet-18 architecture trained on the Kaggle Bone Fracture Dataset, the model can distinguish between fractured and non-fractured bones with high accuracy.

### Key Features
- **Real-time prediction** on uploaded X-ray images
- **Interactive web interface** built with Streamlit
- **Confidence scores** for each prediction
- **Professional medical-style reporting**
- **Mobile-friendly responsive design**

## Model Performance
- **Architecture**: ResNet-18 (modified for grayscale input)
- **Dataset**: Kaggle Bone Fracture Multiregion Dataset
- **Accuracy**: ~92% on test data
- **Classes**: Fractured, Not Fractured

## Tech Stack
- **Deep Learning**: PyTorch, TorchVision
- **Web App**: Streamlit
- **Image Processing**: PIL, OpenCV
- **Data Visualization**: Matplotlib
- **Deployment**: Streamlit Community Cloud

## Project Structure
```
bone-fracture-detector/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── bone_fracture_model.pth   # Trained model weights
├── notebooks/
│   └── training.ipynb        # Model training notebook
├── sample_images/            # Demo images
└── README.md
```

## Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bone-fracture-detector.git
   cd bone-fracture-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## Deployment Guide

### Streamlit Community Cloud (Recommended)

1. **Push to GitHub**
   - Create a public repository
   - Upload all files including `bone_fracture_model.pth`
   - If model file is large, use Git LFS

2. **Deploy to Streamlit**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Click "Deploy"

3. **Access your app**
   - Your app will be available at `https://yourapp.streamlit.app`

## How It Works

1. **Image Upload**: Users upload X-ray images through the web interface
2. **Preprocessing**: Images are resized to 224x224, converted to grayscale, and normalized
3. **Prediction**: ResNet-18 model processes the image and outputs probabilities
4. **Results**: Users see fracture detection results with confidence scores
5. **Recommendations**: Medical-style recommendations based on predictions

## Important Disclaimers

- **Educational Purpose Only**: This application is designed for educational and demonstration purposes
- **Not Medical Advice**: Results should never replace professional medical consultation
- **Accuracy Limitations**: While the model shows good performance, it may not detect all types of fractures
- **Professional Consultation Required**: Always consult healthcare professionals for medical diagnosis

## Future Enhancements

- [ ] Support for multiple bone types (arm, leg, spine, etc.)
- [ ] Integration with DICOM medical imaging format
- [ ] Batch processing for multiple images
- [ ] Export reports as PDF
- [ ] Model interpretability with grad-CAM visualizations

## Model Training Details

The model was trained using:
- **Base Architecture**: ResNet-18 pre-trained on ImageNet
- **Modifications**: 
  - First convolutional layer adapted for single-channel (grayscale) input
  - Final fully connected layer modified for binary classification
- **Training Strategy**: Transfer learning with fine-tuning
- **Optimization**: Adam optimizer with learning rate 0.001
- **Training Duration**: 10 epochs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Your Name**  
Email: your.email@example.com  
LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)  
GitHub: [Your GitHub Profile](https://github.com/yourusername)

---

**If you found this project useful, please give it a star!**
