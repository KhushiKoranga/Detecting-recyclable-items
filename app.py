import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load model
model = tf.keras.models.load_model("model/model.h5")

# Classes 
classes = ['glass', 'metal', 'paper', 'plastic', 'trash']


st.title("♻️ Smart Waste Classifier")
st.write("Detect material + recyclability (Upload or Live Camera)")


st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Recyclability logic
    if predicted_class == "trash":
        recycle_status = "❌ Not Recyclable"
        st.error(f"Material: {predicted_class}")
    else:
        recycle_status = "♻️ Recyclable"
        st.success(f"Material: {predicted_class}")

    st.info(f"Status: {recycle_status}")
    st.write(f"Confidence: {confidence:.2f}%")

    if confidence < 60:
        st.warning("⚠️ Low confidence prediction")


st.subheader("🎥 Live Camera")

run = st.checkbox("Start Camera")
stop = st.button("Stop Camera")

FRAME_WINDOW = st.image([])

camera = None

if run:
    camera = cv2.VideoCapture(0)

    while True:
        if stop:
            break

        ret, frame = camera.read()
        if not ret:
            break

        img = cv2.resize(frame, (224, 224))
        img_array = img / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        label = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        
        if label == "trash":
            status = "Not Recyclable"
            color = (0, 0, 255)  # red
        else:
            status = "Recyclable"
            color = (0, 255, 0)  # green

        
        cv2.putText(frame, f"{label} ({confidence:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    camera.release()