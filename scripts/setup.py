"""
Main script for setting up project and training model, returns a trained resnet 50 model
"""
import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import model as md
import dataloader as dl


def main():
    # directory to raw images split into training and validation sets. approximately 20% of the original train data was used to make validation set
    data_dir= 'space-images/data/raw/images' 

    #create dataloader using Dataloader class, takes in data directory and batch size
    batch_size = 4
    print("------------------------")
    print("Creating Dataloaders...")
    d = dl.Dataloader(data_dir, batch_size)

    #plot the images in the training set
    print("------------------------")
    print("Plotting batch of images...")
    d.plot_images() 

    # instantiate pre-trained resnet 50
    resnet = md.ResNet(batch_size, d.dataset_sizes)

    # train model
    num_epochs = 10
    print("------------------------")
    print("Training model...")
    resnet.train_model(d.dataloaders, num_epochs)
    print("------------------------")
    print("Visualizing results...")
    resnet.visualize_results(d.val_loader)
    print("------------------------")
    print("Saving model")
    resnet.savemodel()

if __name__ == '__main__':
    main()