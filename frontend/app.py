import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2, os

# ==============================
# Load Model
# ==============================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "best_unet_model_v1.h5")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)



model = load_model()

# ==============================
# Preprocessing
# ==============================
def preprocess_image(image):
    img = image.resize((256, 256))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ==============================
# Predict Mask
# ==============================
def predict_mask(model, image_array, threshold=0.5):
    pred = model.predict(image_array)[0]
    mask = (pred > threshold).astype(np.uint8) * 255
    return mask

# ==============================
# Overlay Helper
# ==============================
def create_overlay(image, mask, opacity=0.5, color="green"):
    overlay = image.copy()
    if color == "green":
        overlay[mask > 0] = (0, 255, 0)  # green
    elif color == "blue":
        overlay[mask > 0] = (0, 0, 255)  # red/blue-ish
    else:
        overlay[mask > 0] = (255, 0, 0)  # blue
    return cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)

# ==============================
# Enhance Visibility
# ==============================
def enhance_mask_visibility(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(new_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return new_mask

# ==============================
# Streamlit App
# ==============================
def main():
    st.title("🩺 Polyp Segmentation with U-Net")
    st.write("Upload an endoscopic image to see the polyp segmentation results.")

    # Sidebar Controls
    st.sidebar.header("⚙️ Segmentation Settings")
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.05)
    mask_opacity = st.sidebar.slider("Mask Opacity", 0.1, 1.0, 0.5, 0.05)
    enhance_visibility = st.sidebar.checkbox("Enhance Visibility (remove noise)", value=True)
    min_area = st.sidebar.slider("Min Polyp Area (px)", 100, 5000, 500, 100)

    uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open Image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width="stretch")

        # Preprocess & Predict
        img_array = preprocess_image(image)
        mask = predict_mask(model, img_array, threshold)

        # Resize outputs
        orig_resized = np.array(image.resize((256, 256)))
        mask_resized = mask.squeeze()

        # Enhance mask if selected
        if enhance_visibility:
            mask_resized = enhance_mask_visibility(mask_resized, min_area)

        # Create Binary Mask (white polyp on black bg)
        binary_mask_bw = np.zeros_like(mask_resized)       # start with black background
        binary_mask_bw[mask_resized > 0] = 255            # set polyp area to white


        # Create Overlay
        overlay = create_overlay(orig_resized, mask_resized, mask_opacity, color="green")

        # Display Results (3 Panels)
        st.subheader("🔍 Segmentation Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(orig_resized, caption="Original Image", width="stretch")
        with col2:
            st.image(binary_mask_bw, caption="Binary Mask (Black=Background, White=Polyp)", width="stretch", clamp=True)
        with col3:
            st.image(overlay, caption="Overlay (Green Highlight)", width="stretch")

# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    main()
