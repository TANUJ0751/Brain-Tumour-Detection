import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import gdown
import os

# Google Drive file ID for custom model
FILE_ID = "1x-iy7j1yqidUh5ctwr6dsF7w9-05y9HN"
CUSTOM_MODEL_PATH = "Custom_model.h5"

# ----------------- DOWNLOAD MODEL FROM DRIVE -----------------
if not os.path.exists(CUSTOM_MODEL_PATH):
    st.info("Downloading Custom CNN model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, CUSTOM_MODEL_PATH, quiet=False)




# ----------------- CONFIG -----------------
IMG_SIZE = (224, 224)  # VGG19/ResNet use 224x224

# Load class names used during training

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# ----------------- LOAD MODELS -----------------
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI image and choose a model to predict tumor type.")

# Model selection
model_option = st.pills("Choose a Model",["Tanujs CNN Model", "VGG19", "ResNet50"],default="Tanujs CNN Model")


# Map model to path
MODEL_PATHS = {
    "Tanujs CNN Model": "Custom_model.h5",
    "VGG19": "vgg19_model.h5",
    "ResNet50": "resnet_model.h5"
}

model = load_model(MODEL_PATHS[model_option])

# ----------------- UPLOAD IMAGE -----------------
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file).convert("RGB")


    # Preprocess image
    img = image.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)

    if model_option == "Tanujs CNN Model":
        # No manual normalization, as the model already has Rescaling layer
        img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)
    elif model_option == "VGG19":
        img_array = vgg_preprocess(img_array)
        img_array = np.expand_dims(img_array, axis=0)
    elif model_option == "ResNet50":
        img_array = resnet_preprocess(img_array)
        img_array = np.expand_dims(img_array, axis=0)

    col1, col2 = st.columns(2)

    # Display image in left column
    with col1:
        st.image(image, caption="Uploaded MRI", use_container_width=True)

    # Prediction in right column
    with col2:
        st.subheader("Prediction")
        
        prediction = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction) * 100)

        st.markdown(f"<h2 style='text-align:left; color:#2E8B57;'> {predicted_class}</h2>",unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:left; color:#2E8B57;'> Confidence: {confidence:.2f}%</h3>",unsafe_allow_html=True)
        st.subheader("Class Probabilities:")
        probabilities = " | ".join([f"{cls}: {prediction[0][i]*100:.2f}%" for i, cls in enumerate(CLASS_NAMES)])
        st.markdown(f"{probabilities}")
