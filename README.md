# Space Images: Real or Fake?

## Description

## Run Streamlit app from cloud
The front-end application is available on 
* [Google Cloud](https://second-zephyr-358401.uc.r.appspot.com/) 
* [Streamlit Cloud](https://diarrabell-space-images-app-112ad7.streamlitapp.com/)

## Build Streamlit app locally



## Training Model
The scripts used for training the model can be found in the <code>scripts</code> folder. I recommend training the model in Colab using GPU.
### Train model using Google Colab
Open <code>notebooks/space.ipynb</code> in Colab and run all cells using GPU. The program will ask for your Google account info to download the training dataset. Once all cells have finished, download the newly trained model <code>models/resnetmodel.pt</code>.
### Train model locally 
 <code>setup.py</code> contains a main function that builds datasets, creates dataloaders, and trains the model.
1. Download the training data <code>images.zip</code> from Google Drive [here](https://drive.google.com/file/d/10C-0jNSiH-dGXnQ8XqIBH2I1VvIOd_sZ/view?usp=sharing). Unzip the files and move <code>images</code> folder into <code>data/raw</code>.
2. Run <code>scripts/setup.py</code>. The newly trained model will upload to the models folder.





