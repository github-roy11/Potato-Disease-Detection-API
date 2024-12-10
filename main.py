from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# Configure CORS for specific origins
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the model path and check for existence
MODEL_PATH = "../saved_models/1"
if not os.path.exists(MODEL_PATH) or not os.path.isfile(os.path.join(MODEL_PATH, "saved_model.pb")):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Ensure the model directory exists and contains the saved_model.pb file.")

# Load the model using TFSMLayer or SavedModel format if possible
try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)  # Attempt loading as a standard Keras model
except ValueError as e:
    # Use TFSMLayer if Keras model loading fails
    from tensorflow.keras import layers

    MODEL = layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')

# Define class names for prediction output
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    # Read the uploaded image file
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # Add batch dimension

    # Perform prediction
    predictions = MODEL(img_batch, training=False) if hasattr(MODEL, '__call__') else MODEL.predict(img_batch)

    # Extract predicted class and confidence
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
