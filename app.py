import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="AI Image Classifier", layout="centered")

# 2. Load the Model
with st.spinner('Loading model...'):
    model = load_model('best_model.h5')

# 3. UI Header
st.title("🌼 vs 🌻 Flower Classifier")
st.write("Upload an image to classify it using EfficientNetB0.")

# 4. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 # Normalize
    
    # Predict
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    
    # Interpret Result
    if confidence > 0.5:
        label = "dandelion"
        score = confidence * 100
    else:
        label = "sunflower"
        score = (1 - confidence) * 100
        
    st.success(f"Prediction: **{label}**")
    st.write(f"Confidence: {score:.2f}%")