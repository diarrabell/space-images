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
    def __init__(self, model, dataloader, device, class_names) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.class_names = class_names
        
    def test_model(self):
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
        return self.test_preds

    