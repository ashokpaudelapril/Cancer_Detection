import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('model.keras')
class_names = ['benign', 'malignant', 'normal']

# Title
st.title("Cancer Image Classifier")
st.write("Upload an image to detect if it's benign, malignant, or normal.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((150, 150))  # same size used during training
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: **{pred_class.upper()}**")
