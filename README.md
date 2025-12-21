# Brain Tumor Detection using Machine Learning

A Machine Learning project to detect **brain tumors** from MRI images.  
Includes a **Streamlit web app** where users can upload MRI scans and get predictions for tumor type: **Glioma, Meningioma, Pituitary**, or **No Tumor**.

## Features

- Predict brain tumor type from MRI images.
- User-friendly Streamlit interface.
- Real-time predictions.
- Deployment-ready.

## Project Structure

BrainTumor/
├─ app.py # Streamlit app
├─ trained_model.pkl # Pre-trained ML model
├─ Requirements.txt # Python dependencies
├─ README.md # Project description
└─ utils.py # Helper scripts (optional)

## Installation

1. Clone the repository:
git clone https://github.com/Nishuyeole12/brain-tumor-detection-ml.git
cd brain-tumor-detection-ml

2. Install dependencies:

pip install -r Requirements.txt

**How to Run**

streamlit run app.py

Open the URL shown in terminal.
Upload an MRI image and get tumor predictions instantly.

**Technologies Used**
Python

Scikit-learn / TensorFlow / Keras

Streamlit

NumPy, Pandas, Matplotlib

