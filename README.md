# Space Images: Real or Fake?

## Description
This app uses a ResNet 18 model to classify astronomy-related images as either "real" (captured by a telescope) or "fake" (entirely computer generated).

This project was inspired by the following project led by researchers at the University of Hertfordshire: [Realistic galaxy image simulation via score-based generative models](https://arxiv.org/pdf/2111.01713.pdf). The paper, which was published in February 2022, details how the authors created realistic computer-generated images of galaxies and other space objects using machine learning. 

These images, known as synthetic data, are intended to be used in the field of astronomy to test theories and predict the evolution of galaxies. The authors trained a machine learning model on images from [NASA's Astronomy Picture of the Day (APOD)](https://apod.nasa.gov/apod/astropix.html) archive and created fake APODs that do not actually exist. It would be interesting to see if a machine learning model could differentiate between the raw telescope images and the articfically generated images in order to improve their usage in the field of astronomy. 

This model was trained on images collected from the Hubble and James Webb Telescopes as well as the computer generated images from the research paper.

## Run Streamlit app from cloud
The frontend application is available on:
* [Google Cloud](https://second-zephyr-358401.uc.r.appspot.com/) 
* [Streamlit Cloud](https://diarrabell-space-images-app-112ad7.streamlitapp.com/)

To get started, simply upload an image and wait for the prediction!

## Build Streamlit app locally
In the directory that the app is saved, run the following in the terminal:
<code>make run</code>

To run the app in a Docker container:
<code>make run-container</code>

## Training Model
The scripts used for training the model can be found in the <code>scripts</code> folder. I recommend training the model in Colab using GPU.
### Train model using Google Colab
Open <code>notebooks/space.ipynb</code> in Colab and run all cells using GPU. The program will ask for your Google account info to download the training dataset. Once all cells have finished, download the newly trained model <code>models/resnetmodel.pt</code>.
### Train model locally 
 
1. Download the training data <code>images.zip</code> from Google Drive [here](https://drive.google.com/file/d/10C-0jNSiH-dGXnQ8XqIBH2I1VvIOd_sZ/view?usp=sharing). Unzip the files and move <code>images</code> folder into <code>data/raw</code>.
2. Run <code>scripts/setup.py</code>. The newly trained model will upload to the models folder.

## Description of Scripts
* <code>app.py</code>: script that runs Streamlit frontend app. This script uses a trained model to predict the authenticity of a user-uploaded image.
* <code>test.py</code>: contains helper function for <code>app.py</code>, generates predictions for test data
* <code>setup.py</code>: contains a main function that builds datasets, creates dataloaders, and trains the model
* <code>dataloader.py</code>: creates datasets and dataloaders from training set
* <code>model.py</code>: instantiates and trains model

## Data Sources
* [Kaggle: Top 100 Hubble Telescope Images](https://www.kaggle.com/datasets/redwankarimsony/top-100-hubble-telescope-images)
* [Kaggle: James Webb Telescope Images](https://www.kaggle.com/datasets/goelyash/james-webb-telescope-images-original-size)
* [This is not an APOD](http://www.mjjsmith.com/thisisnotanapod/)
* [This is not a galaxy](http://www.mjjsmith.com/thisisnotagalaxy/)

## Additional References
* https://twitter.com/FakeAstropix/
* https://apod.nasa.gov/apod/astropix.html
* Smith, Michael J., Geach, James E., Jackson, Ryan A., Arora, Nikhil, Stone, Connor, and Courteau, Stéphane. Realistic galaxy image simulation via score-based generative models. United Kingdom: N. p., 2022. Web. doi:10.1093/mnras/stac130. https://arxiv.org/pdf/2111.01713.pdf
