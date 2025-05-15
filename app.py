import streamlit as st
import pandas as pd
import numpy as np
import cv2
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Color Detector", layout="centered")

st.title("🎨 Color Detector App")

uploaded_file = st.file_uploader("Upload an image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

@st.cache_data
def load_colors():
    try:
        df = pd.read_csv("colors.csv")
        if not all(col in df.columns for col in ["color_name", "R", "G", "B"]):
            st.error("colors.csv is missing required columns.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading colors.csv: {e}")
        return None

colors_df = load_colors()

if uploaded_file is not None and colors_df is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Invalid image file.")
        st.stop()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img_rgb.shape

    st.markdown("### Uploaded Image (click on the image to detect color)")
    coords = streamlit_image_coordinates(img_rgb, key="image_coords")

    if coords:
        x, y = coords["x"], coords["y"]
        if 0 <= x < width and 0 <= y < height:
            pixel_rgb = img_rgb[y, x]
            r, g, b = int(pixel_rgb[0]), int(pixel_rgb[1]), int(pixel_rgb[2])

            color_vals = colors_df[["R", "G", "B"]].values
            distances = np.sqrt(np.sum((color_vals - np.array([r, g, b]))**2, axis=1))
            min_idx = np.argmin(distances)
            closest_color = colors_df.iloc[min_idx]

            color_name = closest_color["color_name"]
            cr, cg, cb = int(closest_color["R"]), int(closest_color["G"]), int(closest_color["B"])
            hex_code = f"#{cr:02X}{cg:02X}{cb:02X}"

            st.markdown(f"**Detected Color:** {color_name}")
            st.markdown(f"**RGB:** ({cr}, {cg}, {cb})")
            st.markdown(f"**HEX:** {hex_code}")

            st.markdown(
                f"<div style='width:100px; height:100px; background-color:{hex_code}; border:1px solid #000;'></div>",
                unsafe_allow_html=True
            )
        else:
            st.warning("Clicked outside image bounds.")
    else:
        st.info("Click on the image to detect color.")
elif uploaded_file is None:
    st.info("Please upload an image file to start.")
elif colors_df is None:
    st.error("colors.csv file is missing or invalid. Please ensure it is present i
