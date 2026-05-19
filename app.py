import streamlit as st

st.title("Handwriting Tool")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)