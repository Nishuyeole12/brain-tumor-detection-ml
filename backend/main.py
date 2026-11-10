
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
import os

# Initialize FastAPI
app = FastAPI(
    title="Brain Tumor Detection API",
    description="AI-powered Brain MRI Classification API",
    version="1.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_model.h5")

class_names = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]

# Load trained model
try:
    model = load_model(MODEL_PATH)
    print("Brain Tumor Model Loaded Successfully!")

except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")


# Home API
@app.get("/")
def home():
    return {
        "success": True,
        "message": "Brain Tumor Detection API is running"
    }


# Prediction API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported."
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    try:
        # Read uploaded image and convert to RGB
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize image
        img = img.resize((224, 224))

        # Convert to NumPy array
        img_array = np.array(img, dtype=np.float32)

        # Normalize pixels
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Model prediction
        predictions = model.predict(img_array, verbose=0)

        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)

        # Class probabilities
        probabilities = {
            class_names[i]: round(
                float(predictions[0][i]) * 100, 2
            )
            for i in range(len(class_names))
        }

        return {
            "success": True,
            "prediction": class_names[predicted_class],
            "confidence": round(confidence, 2),
            "probabilities": probabilities
        }

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
