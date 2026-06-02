import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Flower Classifier",
    layout="centered"
)

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_my_model()

st.title("🌼 Dandelion vs 🌻 Sunflower Classifier")

uploaded_file = st.file_uploader(
    "Upload Flower Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((224, 224))

    img_array = np.array(img).astype("float32")
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = float(model.predict(img_array, verbose=0)[0][0])

    if 0.45 <= prediction <= 0.55:

        st.warning("⚠ Model is uncertain.")

        if prediction >= 0.5:
            label = "Possibly Sunflower"
        else:
            label = "Possibly Dandelion"

        confidence = abs(prediction - 0.5) * 200

    elif prediction > 0.5:

        label = "Sunflower"
        confidence = prediction * 100

    else:

        label = "Dandelion"
        confidence = (1 - prediction) * 100

    st.success(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    st.progress(min(confidence / 100, 1.0))