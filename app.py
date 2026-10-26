import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os

st.set_page_config(page_title="Signature Verification", page_icon="✍️", layout="centered")
st.title("✍️ Handwritten Signature Verification")
st.caption("B.Tech Mini Project | Siamese Neural Network + Image Similarity Demo")
st.info(
    "This is an educational prototype, not a real banking/security system. "
    "A similarity score alone cannot prove that a signature is genuine."
)

def prepare_image(uploaded_file, size=(155, 220)):
    img = Image.open(uploaded_file).convert("L")
    img = ImageOps.autocontrast(img)
    img = ImageOps.pad(img, size, color=255)
    return img

def similarity_score(img1, img2):
    """Simple normalized pixel-distance baseline for the app demo."""
    a = np.asarray(img1, dtype=np.float32) / 255.0
    b = np.asarray(img2, dtype=np.float32) / 255.0
    # Remove much of the white background's influence by focusing on darker pixels.
    ink_a = 1.0 - a
    ink_b = 1.0 - b
    mae = np.mean(np.abs(ink_a - ink_b))
    return float(np.clip(1.0 - mae, 0.0, 1.0))

st.markdown("### 1. Upload a reference signature and a questioned signature")
left, right = st.columns(2)
with left:
    ref_file = st.file_uploader("Reference / genuine signature", type=["png", "jpg", "jpeg"], key="ref")
with right:
    test_file = st.file_uploader("Signature to compare", type=["png", "jpg", "jpeg"], key="test")

threshold = st.slider("Demo similarity threshold", min_value=0.30, max_value=0.95, value=0.70, step=0.01)

if ref_file and test_file:
    ref_img = prepare_image(ref_file)
    test_img = prepare_image(test_file)
    c1, c2 = st.columns(2)
    with c1:
        st.image(ref_img, caption="Reference signature", use_container_width=True)
    with c2:
        st.image(test_img, caption="Questioned signature", use_container_width=True)

    if st.button("Compare signatures", type="primary", use_container_width=True):
        score = similarity_score(ref_img, test_img)
        st.metric("Pixel similarity score (demo baseline)", f"{score * 100:.1f}%")
        if score >= threshold:
            st.success("The images look similar under this simple demo threshold.")
        else:
            st.warning("The images look different under this simple demo threshold.")
        st.caption(
            "Important: the current Streamlit demo uses a simple image-similarity baseline. "
            "Train the Siamese model using train.py and a labelled signature dataset before "
            "claiming Siamese-model results. The threshold must be validated on held-out data."
        )
else:
    st.write("Upload two signature images to run a comparison.")

with st.expander("Project details"):
    st.write("""
    **Objective:** Compare two handwritten signature images and estimate whether they are similar.

    **Planned deep-learning model:** A Siamese CNN processes both images with shared weights,
    creates feature vectors, and learns a distance between them using labelled genuine/forged pairs.

    **Suggested evaluation:** Accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
    Use separate training, validation, and test writers/signatures to reduce data leakage.
    """)
