"""
This script trains the model and makes predictions
"""

import os
import dataloader as DL
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

MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'models'))
 #set random seeds for reproducibility
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

class ResNet:
    def __init__(self, batch_size, dataset_sizes) -> None: #initialize resnet18 model
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        # instantiate pre-trained resnet
        self.model = torchvision.models.resnet18(pretrained=True)
        # shut off autograd for all layers to freeze model so layer weights are not trained
        for param in self.model.parameters():
            param.requires_grad = False       
        # get number of inputs to final linear layer
        self.num_ftrs = self.model.fc.in_features
        # replace final linear layer with new linear with 2 outputs for both classes
        self.model.fc = nn.Linear(self.num_ftrs, 2)
        # cost function: cross entropy loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer: SGD
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(self, dataloaders, num_epochs): #train model
        model = self.model.to(self.device)
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train': #set model to training mode
                    model.train()
                else:
                    model.eval() # set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # get input images and labels, and send to GPU if available
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    #zero the weight gradients
                    self.optimizer.zero_grad()

                    #forward pass to get outputs and calculate loss
                    #track gradient only for training data
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        #backpropagation to get the gradients with respect to each weight, only if in train
                        if phase == 'train':
                            loss.backward()
                            # update the weights
                            self.optimizer.step()

                    # convert loss into a scalar and add it to running loss
                    running_loss += loss.item() * inputs.size(0)
                    #track number of correct predictions
                    running_corrects += torch.sum(preds == labels.data)

                # step along learning rate scheduler when in train
                if phase == 'train':
                    self.lr_scheduler.step()
                
                # calculate and display average loss and accuracy for the epoch
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # if model performs better on val set, save weights as the best model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            print()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:3f}'.format(best_acc))

        #load weights from best model
        model.load_state_dict(best_model_wts)
        self.final_model = model

    def savemodel(self): #save model in models folder
        filename = 'resnetmodel.pt'
        path = os.path.join(MODELS_PATH, filename)
        torch.save(self.final_model, path)
    
    def visualize_results(self, val_loader, class_names):
        model = self.final_model.to(self.device)
        with torch.no_grad():
            model.eval()
            # get a batch of validation images
            images, labels = iter(val_loader).next()
            images, labels = images.to(self.device), labels.to(self.device)
            # get predictions
            _, preds = torch.max(model(images), 1)
            preds = np.squeeze(preds.cpu().numpy())
            images = images.cpu().numpy()

        # plot the images in the batch, along with predicted and true labels
        fig= plt.figure(figsize=(15, 10))
        for idx in np.arange(len(preds)):
            ax = fig.add_subplot(2, len(preds)//2, idx+1, xticks=[], yticks=[])
            image = images[idx]
            image = image.transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)
            ax.imshow(image)
            ax.set_title("{} ({})".format(class_names[preds[idx]], class_names[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx] else "red"))
        plt.show()

def test_model(model, test_loader, device):
    model = model.to(device)
    #turn autograd off
    with torch.no_grad():
        #set the model to evaluation mode
        model.eval()
        #set up lists to store true and predicted values
        y_true = []
        test_preds = []
        test_probs = []
        #calculate the predictions on the test set and add to list
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            #feed inputs through model to get raw scores
            logits = model.forward(inputs)
            #convert raw scores to probabilities
            probs = F.softmax(logits, dim=1)
            #get discrete predictions using argmax
            preds = np.argmax(probs.numpy(), axis=1)
            #add predictions and actuals to lists
            test_preds.extend(preds)
            test_probs.extend(probs)
            y_true.extend(labels)
        
        #calculate the accuracy
        # test_preds = np.array(test_preds)
        # test_probs = np.array(test_probs)
        # y_true = np.array(y_true)
        # test_acc = np.sum(test_preds == y_true)/y_true.shape[0]

        #recall for each class
        # recall_vals = []
        # for i in range(2):
        #     class_idx = np.argwhere(y_true==i)
        #     total = len(class_idx)
        #     correct = np.sum(test_preds[class_idx]==i)
        #     recall = correct/total
        #     recall_vals.append(recall)
    
    return test_preds, test_probs

