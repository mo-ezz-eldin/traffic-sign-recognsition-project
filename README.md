🚦 AI Traffic Sign Recognition
An intelligent traffic sign recognition system built with deep learning that achieves 99.91% accuracy in classifying 43 different types of traffic signs. The application features a beautiful Streamlit web interface for real-time predictions.

🌟 Features

High Accuracy: 99.91% classification accuracy on test data
Real-time Prediction: Instant traffic sign recognition
Beautiful UI: Modern, responsive web interface with glassmorphism design
43 Sign Types: Comprehensive coverage of traffic signs
Confidence Scores: Detailed prediction confidence with top-3 results
Easy to Use: Simple drag-and-drop image upload

🚀 Quick Start
Prerequisites
bashPython 3.8+
PyTorch
Streamlit
PIL (Pillow)
NumPy
Installation

Clone the repository

bashgit clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition

Install dependencies

bashpip install -r requirements.txt

Download the trained model

The model file should be named traffic_sign_model.pth
Place it in the project root directory
Download from: [Model Download Link]


Run the application

bashstreamlit run streamlit_app.py

Open your browser

Navigate to http://localhost:8501
Upload a traffic sign image and get instant predictions!



📁 Project Structure
traffic-sign-recognition/
│
├── streamlit_app.py           # Main Streamlit application
├── traffic_sign_model.pth     # Trained PyTorch model weights
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── notebooks/                 # Jupyter notebooks
│   └── Traffic_Sign_Training.ipynb  # Google Colab training notebook
├── data/                      # Dataset (if included)
├── images/                    # Sample images for testing
└── docs/                      # Additional documentation
🧠 Model Architecture

Base Model: ResNet-18 (pre-trained on ImageNet)
Fine-tuning: Custom fully connected layer for 43 traffic sign classes
Input Size: 224x224 RGB images
Training: Transfer learning approach for optimal performance

Model Performance
MetricScoreAccuracy99.91%Precision99.89%Recall99.91%F1-Score99.90%
🎯 Supported Traffic Signs
The model can recognize 43 different types of traffic signs, including:

Speed Limits: 20km/h, 30km/h, 50km/h, 60km/h, 70km/h, 80km/h, 100km/h, 120km/h
Warning Signs: Dangerous curves, bumpy road, slippery road, pedestrians, children crossing
Regulatory Signs: Stop, yield, no entry, no passing, priority road
Mandatory Signs: Go straight, turn right/left, roundabout, keep right/left
And many more...

🔧 Technical Details
Image Preprocessing
pythontransforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
Prediction Pipeline

Image Upload: User uploads traffic sign image
Preprocessing: Resize, normalize, and convert to tensor
Model Inference: Forward pass through ResNet-18
Post-processing: Apply softmax for confidence scores
Results Display: Show prediction with confidence metrics

📊 Training Details
The model was trained using Google Colab with the following specifications:

Dataset: German Traffic Sign Recognition Benchmark (GTSRB)
Training Images: ~39,000 images
Test Images: ~12,000 images
Epochs: 20
Optimizer: Adam
Learning Rate: 0.001
Batch Size: 32

Training Notebook
📓 Google Colab Notebook: Traffic Sign Training Notebook
🎨 User Interface
The Streamlit application features:

Modern Design: Glassmorphism UI with gradient backgrounds
Responsive Layout: Works on desktop and mobile
Interactive Elements: Hover effects and smooth animations
Real-time Feedback: Instant prediction results
Confidence Visualization: Progress bars and confidence metrics

📱 Usage Examples
Command Line Prediction
pythonimport torch
from PIL import Image
from torchvision import transforms

# Load model and predict
model = load_model()
image = Image.open('traffic_sign.jpg')
prediction = predict(model, preprocess_image(image))
print(f"Prediction: {CLASS_NAMES[prediction[0]]}")
print(f"Confidence: {prediction[1]:.2%}")
Web Interface

Open the Streamlit app
Upload a traffic sign image
View instant predictions with confidence scores
Explore top-3 predictions for detailed analysis

🛠️ Development
Setting up Development Environment

Fork the repository
Create a virtual environment

bashpython -m venv traffic_sign_env
source traffic_sign_env/bin/activate  # On Windows: traffic_sign_env\Scripts\activate

Install development dependencies

bashpip install -r requirements.txt
Testing
bash# Test with sample images
python test_model.py

# Run the Streamlit app locally
streamlit run streamlit_app.py
📈 Performance Optimization

Model Size: Optimized for fast inference (~45MB)
CPU Inference: Runs efficiently on CPU for deployment
Memory Usage: Low memory footprint for edge devices
Response Time: < 1 second prediction time

🚀 Deployment Options
Local Deployment
bashstreamlit run streamlit_app.py
📄 Requirements
Create a requirements.txt file with:
txtstreamlit>=1.28.0
torch>=1.13.0
torchvision>=0.14.0
Pillow>=9.0.0
numpy>=1.21.0
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
👨‍💻 Author
Your Name

GitHub: @yourusername
LinkedIn: Your LinkedIn
Email: your.email@example.com

🙏 Acknowledgments

German Traffic Sign Recognition Benchmark (GTSRB) for the dataset
PyTorch team for the deep learning framework
Streamlit for the amazing web app framework
ResNet authors for the base architecture

📊 Dataset Information
The model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset:

Total Images: ~50,000 images
Classes: 43 traffic sign categories
Resolution: Variable (resized to 224x224 for training)
Format: RGB images

Class Distribution
The dataset includes balanced representation of:

Speed limit signs (8 categories)
Warning signs (11 categories)
Mandatory signs (5 categories)
Prohibitory signs (19 categories)
