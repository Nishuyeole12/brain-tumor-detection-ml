import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = load_model("brain_tumor_model.h5")

# Class labels (same order as training)
class_names = [
    'Glioma Tumor',
    'Meningioma Tumor',
    'No Tumor',
    'Pituitary Tumor'
]

st.set_page_config(
    page_title="Brain Tumor Detection System",
    layout="centered"
)

# Title
st.title("🧠 Brain Tumor Detection System")
st.markdown(
    "### AI-Based Clinical Decision Support Tool"
)

st.divider()

# Upload MRI Image
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("🔍 Analyze MRI"):
        # Preprocess image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        st.success("✅ Analysis Completed")

        st.subheader("🧾 Prediction Result")
        st.write(f"**Tumor Type:** {class_names[predicted_class]}")
        st.write(f"**Confidence Score:** {confidence:.2f}%")

        # Confidence bar
        st.progress(int(confidence))

st.divider()

# Medical Disclaimer
st.warning(
    "⚠ This system is intended for clinical decision support only. "
    "Final diagnosis must be confirmed by a certified medical professional."
)

st.caption("Developed using Deep Learning (CNN) | MRI Image Analysis")
