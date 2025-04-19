# equation_tool.py

import streamlit as st
from PIL import Image
import numpy as np

def equation_tool_ui():
    st.header("🧮 Image Equation Mapper")

    def parse_equation(eq_str):
        eq_str = eq_str.replace("^", "**")
        if "x" not in eq_str:
            raise ValueError("Equation must contain 'x'")
        def equation(x):
            return eval(eq_str.replace("x", str(x)))
        return equation

    def clamp(val):
        return max(0, min(255, int(val)))

    def apply_equation_channel(channel, func):
        arr = np.array(channel)
        vectorized_func = np.vectorize(lambda x: clamp(func(x)))
        return Image.fromarray(vectorized_func(arr).astype('uint8'))

    def apply_transformations(img, func):
        gray = img.convert("L")
        r, g, b = img.split()

        gray_trans = apply_equation_channel(gray, func)
        r_new = apply_equation_channel(r, func)
        g_new = apply_equation_channel(g, func)
        b_new = apply_equation_channel(b, func)

        rgb_trans = Image.merge("RGB", (r_new, g_new, b_new))

        return {
            "gray": gray,
            "r": r,
            "g": g,
            "b": b,
            "gray_trans": gray_trans,
            "r_trans": r_new,
            "g_trans": g_new,
            "b_trans": b_new,
            "rgb_trans": rgb_trans
        }

    # ===== Main UI =====
    uploaded = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Original Image", use_container_width=True)

        # Show original channels
        gray = image.convert("L")
        r, g, b = image.split()

        st.subheader("📷 Original Channels")
        col1, col2, col3, col4 = st.columns(4)
        col1.image(gray, caption="Grayscale", use_container_width=True)
        col2.image(r, caption="R Channel", use_container_width=True)
        col3.image(g, caption="G Channel", use_container_width=True)
        col4.image(b, caption="B Channel", use_container_width=True)

        # Input equation and apply
        st.subheader("✍️ Apply Equation")
        equation = st.text_input("Enter your equation (use 'x')", value="x^2 / 255", key="equation_input")

        if st.button("Apply Equation"):
            try:
                func = parse_equation(equation)

                # Add the processing spinner here
                with st.spinner('Processing your image...'):
                    result = apply_transformations(image, func)

                st.subheader("✨ Transformed Outputs")
                col5, col6 = st.columns(2)
                col5.image(result["gray_trans"], caption="Grayscale Transformed", use_container_width=True)
                col6.image(result["rgb_trans"], caption="RGB Transformed", use_container_width=True)

                st.subheader("🎨 Transformed Channels")
                col7, col8, col9 = st.columns(3)
                col7.image(result["r_trans"], caption="R Transformed", use_container_width=True)
                col8.image(result["g_trans"], caption="G Transformed", use_container_width=True)
                col9.image(result["b_trans"], caption="B Transformed", use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {e}")