import os
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

st.title("MNIST Digit Predictor")
st.write("Upload an image of a handwritten digit to get a prediction.")

# ปรับชื่อโมเดลให้ตรงกับที่คุณเซฟไว้ (เปลี่ยนเป็น '671010157_mnist_model.keras' ตามรหัสนิสิตของคุณ หรือใช้ชื่อด้านล่าง)
model_path = "671010157_mnist_model.keras"

if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found.")
    st.stop()

try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)

        st.image(
            img,
            caption="Uploaded Image",
            width=300
        )

        # Convert to grayscale
        img = img.convert("L")

        # Resize to MNIST dimensions
        img = img.resize((28, 28))

        # Convert to NumPy array and normalize
        img_array = np.array(img).astype("float32") / 255.0

        # Add channel and batch dimensions:
        # (28, 28) -> (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)

        st.write("Classifying...")

        prediction = model.predict(img_array, verbose=0)
        predicted_digit = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0])) * 100

        st.success(
            f"The model predicts the digit is: **{predicted_digit}**"
        )
        st.write(f"Confidence: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
