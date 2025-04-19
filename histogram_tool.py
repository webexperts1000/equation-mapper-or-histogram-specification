# histogram_tool.py

import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# --- Histogram Functions ---

def compute_histogram(img):
    hist = np.zeros(256, dtype=np.int32)
    for pixel in img.flatten():
        hist[pixel] += 1
    return hist

def compute_cdf(hist):
    return np.cumsum(hist)

def histogram_specification(source, reference):
    src_hist = compute_histogram(source)
    ref_hist = compute_histogram(reference)
    src_cdf = compute_cdf(src_hist)
    ref_cdf = compute_cdf(ref_hist)

    lookup = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        j = np.argmin(np.abs(ref_cdf - src_cdf[i]))
        lookup[i] = j
    return lookup[source].astype(np.uint8)

def match_rgb_channels(src_img, ref_img):
    matched_channels = []
    for i in range(3):
        matched = histogram_specification(src_img[:, :, i], ref_img[:, :, i])
        matched_channels.append(matched)
    return np.stack(matched_channels, axis=2).astype(np.uint8)

# --- Helper Functions ---

def show_image_channels(title, img_np):
    st.markdown(f"### 🖼️ {title} Image Views")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_np, caption="RGB", use_container_width=True)

    with col2:
        gray = np.mean(img_np, axis=2).astype(np.uint8)
        st.image(gray, caption="Grayscale", use_container_width=True)

    with col3:
        ch_titles = ["Red", "Green", "Blue"]
        ch_imgs = [img_np[:, :, i] for i in range(3)]
        st.image(ch_imgs, caption=ch_titles, width=90)

    return gray

def show_histograms(title, img_np, gray_img):
    st.markdown(f"### 📊 {title} Histograms")
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))

    # Grayscale
    gray_hist = compute_histogram(gray_img)
    axs[0].plot(gray_hist, color='gray')
    axs[0].set_title("Grayscale Histogram")

    # RGB
    colors = ['r', 'g', 'b']
    for i, c in enumerate(colors):
        ch_hist = compute_histogram(img_np[:, :, i])
        axs[1].plot(ch_hist, color=c, label=c.upper())

    axs[1].legend()
    axs[1].set_title("RGB Histograms")

    st.pyplot(fig)

# --- Streamlit UI ---

def histogram_tool_ui():
    st.header("📊 Histogram Specification Tool")

    with st.expander("📥 Upload Your Images"):
        col1, col2 = st.columns(2)
        with col1:
            input_file = st.file_uploader("Upload Input Image", type=["png", "jpg", "jpeg"], key="input")
        with col2:
            ref_file = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"], key="ref")

    if input_file and ref_file:
        st.markdown("---")
        st.subheader("🎯 Ready to Process?")
        if st.button("✨ Apply Histogram Specification"):
            with st.spinner("Matching histograms..."):

                # Load images
                src_img = Image.open(input_file).convert("RGB")
                ref_img = Image.open(ref_file).convert("RGB")
                src_np = np.array(src_img)
                ref_np = np.array(ref_img)

                # Convert to grayscale
                gray_src = np.mean(src_np, axis=2).astype(np.uint8)
                gray_ref = np.mean(ref_np, axis=2).astype(np.uint8)

                # Perform histogram matching
                result_rgb = match_rgb_channels(src_np, ref_np)
                result_gray = histogram_specification(gray_src, gray_ref)

                # Output image as PIL
                result_img = Image.fromarray(result_rgb)

                # ========== DISPLAY ALL SECTIONS ==========
                st.markdown("## 🔍 Input Image Analysis")
                show_image_channels("Input", src_np)
                show_histograms("Input", src_np, gray_src)

                st.markdown("## 🧭 Reference Image Analysis")
                show_image_channels("Reference", ref_np)
                show_histograms("Reference", ref_np, gray_ref)

                st.markdown("## 🎉 Output Image Result")
                show_image_channels("Output", result_rgb)
                show_histograms("Output", result_rgb, result_gray)

                st.image(result_img, caption="Final Output Image", use_container_width=True)

                buffered = BytesIO()
                result_img.save(buffered, format="PNG")
                st.download_button("📥 Download Result", data=buffered.getvalue(), file_name="histogram_matched.png", mime="image/png")
