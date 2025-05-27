import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Color Picker dari Gambar", layout="centered")

st.markdown("""
    <style>
        body {
            background: #ffffff;
        }
        .stApp {
            font-family: 'Segoe UI', sans-serif;
        }
        .color-box {
            height: 80px;
            width: 100%;
            border-radius: 8px;
            margin-bottom: 5px;
            border: 1px solid #ccc;
        }
        .hex-text {
            font-family: monospace;
            text-align: center;
            font-size: 15px;
        }
        .title-header {
            font-size: 32px;
            font-weight: bold;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-header'>Color Picker dari Gambar</div>", unsafe_allow_html=True)
st.caption("Ekstraksi palet warna dominan.")

tabs = st.tabs(["Dominant Color", "Tentang Aplikasi"])

with tabs[0]:
    uploaded_file = st.file_uploader("Unggah gambar di sini", type=["jpg", "jpeg", "png"])

    def get_dominant_colors(image, k=5):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (100, 100))
        img = img.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(img)
        return kmeans.cluster_centers_.astype(int)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Pratinjau Gambar", use_column_width=True)

        with st.spinner("Memproses gambar..."):
            colors = get_dominant_colors(image, k=5)

        st.subheader("Palet Warna Dominan")

        hex_colors = []
        cols = st.columns(5)
        for i, col in enumerate(cols):
            rgb = colors[i]
            hex_color = '#%02x%02x%02x' % tuple(rgb)
            hex_colors.append(hex_color)
            with col:
                st.markdown(f"<div class='color-box' style='background-color: {hex_color};'></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='hex-text'>{hex_color}</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("Kode Warna Hex")
        st.code(" ".join(hex_colors), language="text")

with tabs[1]:
    st.header("Tentang Aplikasi")
    st.write("""
    Aplikasi ini menggunakan algoritma **KMeans Clustering** untuk mengekstrak 5 warna paling dominan dari sebuah gambar.
    
    ### Library yang Digunakan:
    - **Streamlit** untuk antarmuka web interaktif
    - **scikit-learn** untuk KMeans clustering
    - **OpenCV** dan **Pillow** untuk pemrosesan gambar
    
    Dibuat dengan tujuan edukatif dan praktis. Bisa dikembangkan lebih lanjut untuk mengunduh palet, integrasi Adobe, dan lainnya.
    """)