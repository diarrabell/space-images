"""
Script that evaluates test data using trained model
"""
import os
import scripts.dataloader 
import scripts.model
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import time
import numpy as np
import pandas as pd
import torch.nn.functional as F
import copy
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

class TestData:
    def __init__(self, model, dataloader, device) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device
        
    def test_model(self, device):
        self.model = self.model.to(self.device)
        #turn autograd off
        with torch.no_grad():
            #set the model to evaluation mode
            self.model.eval()
            #set up lists to store true and predicted values
            y_true = []
            test_preds = []
            test_probs = []
            #calculate the predictions on the test set and add to list
            for data in self.dataloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                #feed inputs through model to get raw scores
                logits = self.model.forward(inputs)
                #convert raw scores to probabilities
                probs = F.softmax(logits, dim=1)
                #get discrete predictions using argmax
                preds = np.argmax(probs.numpy(), axis=1)
                #add predictions and actuals to lists
                test_preds.extend(preds)
                test_probs.extend(probs)
                y_true.extend(labels)

        self.test_preds = test_preds
        self.test_probs = test_probs
        return test_preds

    def visualize_results(self):
        self.model = self.model.to(self.device)
        with torch.no_grad():
            self.model.eval()
            images, labels = iter(self.dataloader).next()
            images, labels = images.to(self.device), labels.to(self.device)
            _,preds = torch.max(self.model(images), 1)
            preds = np.squeeze(preds.cpu().numpy())
            images = images.cpu().numpy()
        fig = plt.figure(figsize=(15, 10))
        for idx in np.arange(len(preds)):
            ax = fig.add_subplot(2, len(preds)//2, idx+1, xticks=[], yticks=[])
            image = images[idx]
            image = image.transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)
            ax.imshow(image)
            ax.set_title("Probability of being Deforested: {0:.3f}, Prediction: {1}".format(self.test_probs[idx][1], preds[idx]))
        return 