"""
In this script you should train your 'clean' weight-space classifier.
"""

import os
import torch
import torch.nn as nn
from utils import set_random_seeds, vec_to_img, get_fmnist_functa
import numpy as np
from SIREN import ModulatedSIREN
import argparse

# A basic linear classifier.
class VanillaClassifier(nn.Module):
    def __init__(self, in_features=512, num_classes=10):
        """
        :param in_features: input_dimension.
        :param num_classes: number of classes (output dimension).
        """
        super(VanillaClassifier, self).__init__()
        self.net = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a weight-space classifier')
    parser.add_argument('-p', '--data-path', type=str, default='/datasets/functaset',
                        help='The path to INR dataset (a.k.a functaset)')
    parser.add_argument('-b', '--batch-size', type=int, default=128,
                        help='The path to INR dataset (a.k.a functaset)')
    parser.add_argument('-c', '--cpu', action='store_true', help = "If set, use cpu and not cuda")
    # add any other parameters you may need here
    args = parser.parse_args()
    
    # Set random seed.
    set_random_seeds(0)
    device = 'cpu' if args.cpu else 'cuda:0'
    
    
   
    # Handle data-loading - these loaders yield(vector,label) pairs.
    train_functaloader = get_fmnist_functa(data_dir=f"{args.data_path}/fmnist_train.pkl",mode='train', batch_size = args.batch_size, num_workers = 2) 
    val_functaloader = get_fmnist_functa(data_dir=f"{args.data_path}/fmnist_val.pkl",mode='test', batch_size = args.batch_size, num_workers = 2)
    test_functaloader = get_fmnist_functa(data_dir=f"{args.data_path}/fmnist_test.pkl",mode='test', batch_size = args.batch_size, num_workers = 2)
    
    # Load Full INR - this is only for visualization purposes - this is just an example, you can erase this when you submit
    inr = ModulatedSIREN(height=28, width=28, hidden_features=256, num_layers=10, modul_features=512)
    inr.load_state_dict(torch.load(f"{args.data_path}/modSiren.pth")['state_dict'])
    inr = inr.to(device)
    
    #Example of extracting full image from modulation vector - must pass a single (non-batched) vector input - this is just an example, you can erase this when you submit
    img = vec_to_img(inr, train_functaloader.dataset[0][0].to(device))
    
    
    # Instantiate Classifier Model
    classifier = VanillaClassifier(in_features=512, num_classes=10).to(device)
    
    #inference example
    predicted_scores = classifier(train_functaloader.dataset[0][0].to(device))
    
    # TODO: Implement your training and evaluation loops here. We recommend you also save classifier weights for next parts
    
  