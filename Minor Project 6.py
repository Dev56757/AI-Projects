import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model("digit_cnn_model.h5")

# Streamlit page configuration
st.set_page_config(page_title="Digit Recognition", page_icon="🔢")

# Title
st.title("🔢 Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (0-9) for prediction.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Open image in grayscale
    image = Image.open(uploaded_file).convert("L")

    # Display original image
    st.subheader("Uploaded Image")
    st.image(image, width=200)

    # Resize image to 28x28
    image = image.resize((28, 28))

    # Convert image to numpy array
    img_array = np.array(image)

    # --------------------------------------------------
    # Automatically invert colors if needed
    # --------------------------------------------------
    # MNIST uses white digits on black background.
    # If uploaded image has black digits on white background,
    # invert the colors.
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Normalize pixel values
    img_array = img_array.astype("float32") / 255.0

    # Display processed image
    st.subheader("Processed Image")
    st.image(img_array, width=150, clamp=True)

    # Reshape for CNN model
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array, verbose=0)

    predicted_digit = np.argmax(prediction[0])

    confidence = np.max(prediction[0]) * 100

    # Display results
    st.success(f"Predicted Digit: {predicted_digit}")

    st.info(f"Confidence: {confidence:.2f}%")