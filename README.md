# Space Images: Real or Fake?

## Description

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
* https://photojournal.jpl.nasa.gov/mission/Hubble+Space+Telescope
* Smith, Michael J., Geach, James E., Jackson, Ryan A., Arora, Nikhil, Stone, Connor, and Courteau, Stéphane. Realistic galaxy image simulation via score-based generative models. United Kingdom: N. p., 2022. Web. doi:10.1093/mnras/stac130. https://arxiv.org/pdf/2111.01713.pdf
