import streamlit as st
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('brain_tumor_model.h5')

# Define a function to make predictions
def predict(image):
    img = load_img(image, target_size=(300, 300))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return prediction[0][0]  # The output is a single value (0 for No, 1 for Yes)

# Streamlit GUI
st.title("Brain Tumor Detection App")

uploaded_image = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    if st.button("Predict"):
        prediction = predict(uploaded_image)
        if prediction > 0.5:
            st.write("Prediction: Yes, it's a brain tumor.")
        else:
            st.write("Prediction: No, it's not a brain tumor.")

st.sidebar.title("About")
st.sidebar.info(
    "This is a simple brain tumor detection app using a trained model. "
    "Upload an MRI image, and the app will predict whether a brain tumor is present or not."
)

