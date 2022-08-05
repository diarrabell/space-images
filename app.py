"""
Script for Streamlit app
"""

import streamlit as st
import test as t
import torch
from torchvision import datasets, transforms
import os
import time
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#function to load trained model
def load_model():
    model = torch.load("models/resnetmodel.pt", map_location=torch.device(device))
    model.eval()
    st.success("loaded model!")
    return model

#function that makes prediction for image based on trained model
def make_prediction(model):
    with st.spinner("Testing image..."):
        time.sleep(5)
        #create dataloader
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_dataset = datasets.ImageFolder("uploads", transform=data_transforms)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        classes = ['fake', 'real']
        test = t.TestData(model, test_dataloader, device, classes)
        prediction = test.test_model()
        idx = int(prediction[0])
        prediction = "predicted class: " + classes[idx]
        st.subheader(prediction)
    return 

#function to load image
def load_image(image):
    return Image.open(image)

#function to load background image: galaxy cluster MACS J0717, taken by Hubble Telescope
def bg_image():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.spacetelescope.org/archives/images/screen/heic1215b.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

#main function to run app
def run():
    bg_image()
    st.title("Determining Authenticity of Space Images")
    st.subheader('**This is an application that predicts whether a space-related image is authentic or computer-generated.**')
    st.markdown("***by diarra bell  \nduke university aipi 540***")
    hide_footer_style = """<style>.reportview-container .main footer {visibility: hidden;}"""
    st.markdown(hide_footer_style, unsafe_allow_html=True)

    #create directory to store images
    if not os.path.exists("uploads/images"):
        os.mkdir("uploads/images")

    #prompt user to upload file to test
    label = "upload your image here"
    uploaded_file = st.file_uploader(label, type=None) 
    if uploaded_file is not None:
        #display image
        st.image(load_image(uploaded_file), caption="filename: {0}, file size: {1}".format(uploaded_file.name, uploaded_file.size))

        #save upload
        with open(os.path.join("uploads/images", uploaded_file.name), "wb") as f:
            f.write((uploaded_file).getbuffer()) 
        st.success("saved file!")

        #load trained model
        model = load_model()
        make_prediction(model)

        #remove file from folder after printing results
        os.remove(os.path.join("uploads/images", uploaded_file.name))

if __name__ == "__main__":
    run()
