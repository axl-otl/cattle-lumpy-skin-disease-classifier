import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

CLASS_NAMES = ["LSD_Infected", "Healthy"]
MODEL_PATH = "cattle_resnet_lsd_model.keras"

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

st.title("Lumpy Skin Disease Classifier")
st.write("Upload a cattle image to classify whether it's Healthy or Infected with Lumpy Skin Disease.")

# Load model at startup (cached, so only runs once)
model = load_trained_model()

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

with col2:
    if uploaded_file is not None:
        with st.spinner("Classifying..."):
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = np.max(predictions)

        st.image(img, caption=f"Classification: {predicted_class} ({confidence:.2%})", use_container_width=True)
        st.write(f"### **Classification: {predicted_class}**")
        st.write(f"Confidence: {confidence:.2%}")
