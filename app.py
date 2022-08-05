"""
Script for Streamlit app
"""

import streamlit as st
import torch
import os
import pandas as pd
import zipfile
from PIL import Image

MODELS_PATH =  os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'models'))
MODEL_FILENAME = 'resnetmodel.pt'
MODEL_PATH = os.path.join(MODELS_PATH, MODEL_FILENAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache
def load_model():
    model = torch.load(MODEL_PATH, map_location=torch.device(device))
    model.eval()
    return model

def make_prediction(image):
    pass

def load_image(image):
    return Image.open(image)

def run():
    st.title("Determining Authenticity of Space Images")
    st.markdown('**This is an application that predicts whether a space-related image is authentic or fabricated.**')
    hide_footer_style = """<style>.reportview-container .main footer {visibility: hidden;}"""
    st.markdown(hide_footer_style, unsafe_allow_html=True)
    
    label = "Upload your image here. Image size is limited to 200 MB."
    uploaded_file = st.file_uploader(label, type=None)
    if uploaded_file is not None:
        st.image(load_image(uploaded_file), width=250)
        st.write("filename:{0} filesize:{1}".format(uploaded_file.name, uploaded_file.size))



if __name__ == "__main__":
    run()
