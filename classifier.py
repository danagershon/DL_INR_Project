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
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


"""
# A basic linear classifier.
class VanillaClassifier(nn.Module):
    def __init__(self, in_features=512, num_classes=10):
        super(VanillaClassifier, self).__init__()
        self.net = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)
"""


class WeightSpaceClassifier(nn.Module):
    """
    Our classifer implementation
    """
    
    def __init__(self, in_features=512, num_classes=10, use_batchnorm=True, dropout_prob=0.2):
        """
        :param in_features: input_dimension (size of modulation vector)
        :param num_classes: number of classes (output dimension). 10 classes, for each type of clothing.
        """
        super(WeightSpaceClassifier, self).__init__()

        layers = [
            # First hidden layer
            nn.Linear(in_features, 512),  
            nn.BatchNorm1d(512) if use_batchnorm else nn.Identity(),
            nn.ReLU(), 

            # Second hidden layer
            nn.Linear(512, 256),  
            nn.ReLU(),  
            nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity(),

            # Third hidden layer
            nn.Linear(256, 128),  
            nn.ReLU(),
            nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity(),

            nn.Linear(128, num_classes)  # Output layer (no activation because CrossEntropyLoss applies softmax internally)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

def train_model(model, train_loader, val_loader, 
                num_epochs=20, learning_rate=0.001, criterion=nn.CrossEntropyLoss(), 
                device='cuda'):
    """
    Train the WeightSpaceClassifier model.

    :param model: The classifier model.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param num_epochs: Number of epochs to train.
    :param learning_rate: Learning rate for the optimizer.
    :param criterion: the loss function
    :param device: Device to train on ('cuda' or 'cpu').

    :return: Trained model.
    """
    model = model.to(device)  # TODO LEFT: remove since this is done outside

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0.0
        correct_predictions = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update model params
            optimizer.step()

            # Add batch loss to the epoch loss
            epoch_loss += loss.item() * inputs.size(0)  # multiply by bach size becuse CE loss returns avg loss over the batch but we want to accumulate loss
            
            # Compute predictions for the batch
            predicted = torch.argmax(outputs, 1)  # no need to apply softmax before max, becuase softmax doesnt change relative ranking
            
            # Add correct batch predicaiton to total correct epoch predictions
            correct_predictions += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for the epoch
        total_samples = len(train_loader.dataset)
        avg_epoch_loss = epoch_loss / total_samples
        epoch_accuracy = (correct_predictions / total_samples) * 100

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Evaluate on the validation set after each epoch
        evaluate_model(model, val_loader, criterion, 'Validation', device)

    return model


def evaluate_model(model, loader, criterion, type, device='cuda'):
    """
    Evaluate the model (on the validation/test set).

    :param model: The classifier model.
    :param val_loader: DataLoader for validation/test data.
    :param criterion: the loss function
    :param type: 'Validatoin' or 'Test' (for printing purposes)
    :param device: Device to evaluate on ('cuda' or 'cpu').
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0

    # Use no_grad to disable gradient calculation during validation
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute batch loss
            loss = criterion(outputs, labels)

            # Accumulate validation loss and correct predictions
            total_loss += loss.item() * inputs.size(0)  # multiply by bach size becuse CE loss returns avg loss over the batch but we want to accumulate loss
            predicted = torch.argmax(outputs, 1)  # no need to apply softmax before max, becuase softmax doesnt change relative ranking
            correct_predictions += (predicted == labels).sum().item()

    # Calculate average validation loss and accuracy
    total_samples = len(loader.dataset)
    total_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples * 100

    print(f"{type} - Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")


def visualize_classifications(model, test_loader, inr_model, num_images=5, device='cuda', output_file='classification_results.png'):
    """
    Visualize 'num_images' correctly classified and 'num_images' incorrectly classified images from the test set.

    :param model: Trained classifier model.
    :param test_loader: DataLoader for test data.
    :param inr_model: Pre-trained ModulatedSIREN model to convert modulation vectors back to images.
    :param num_images: Number of correctly and incorrectly classified images to visualize.
    :param device: Device to run the evaluation ('cuda' or 'cpu').
    :param output_file: File to save the visualization (default: 'classification_results.png').
    """
    model.eval()  # Set the classifier to evaluation mode
    inr_model = inr_model.to(device)  # TODO LEFT: remove since this is done outside
    correct_images = []
    incorrect_images = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, 1)

            # Iterate through each sample in the batch
            for i in range(inputs.size(0)):
                if predicted[i] == labels[i] and len(correct_images) < num_images:
                    correct_images.append((inputs[i], predicted[i], labels[i]))
                elif predicted[i] != labels[i] and len(incorrect_images) < num_images:
                    incorrect_images.append((inputs[i], predicted[i], labels[i]))

                # Stop once we have enough correct and incorrect examples
                if len(correct_images) == num_images and len(incorrect_images) == num_images:
                    break

    # Plot the results
    fig, axs = plt.subplots(2, num_images, figsize=(15, 6))
    fig.suptitle('Correct and Incorrect Classifications')

    for row, images in enumerate([correct_images, incorrect_images]):
        for idx, (vec, pred, label) in enumerate(images):
            img = vec_to_img(inr_model, vec).detach().cpu().numpy()
            axs[row, idx].imshow(img, cmap='gray')
            axs[row, idx].set_title(f"Pred: {pred.item()}, True: {label.item()}")
            axs[row, idx].axis('off')

    # Save the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file)
    print(f"Classification results saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a weight-space classifier')
    parser.add_argument('-p', '--data-path', type=str, default='/datasets/functaset',
                        help='The path to INR dataset (a.k.a functaset)')
    parser.add_argument('-b', '--batch-size', type=int, default=128,
                        help='batch size for the data loader')
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
    
    """
    #Example of extracting full image from modulation vector - must pass a single (non-batched) vector input - this is just an example, you can erase this when you submit
    img = vec_to_img(inr, train_functaloader.dataset[0][0].to(device))
    
    # Instantiate Classifier Model
    classifier = VanillaClassifier(in_features=512, num_classes=10).to(device)
    
    #inference example
    predicted_scores = classifier(train_functaloader.dataset[0][0].to(device))
    """
    
    # TODO: Implement your training and evaluation loops here. We recommend you also save classifier weights for next parts
    
    # --------- Implementation ---------
    
    # initialize model
    model = WeightSpaceClassifier().to(device)

    # define hyperparameters
    NUM_EPOCHS = 20
    LR = 0.001
    CRITERION = nn.CrossEntropyLoss()

    # train model (validation set is evaluated at the end of each epoch)
    train_model(model, train_functaloader, val_functaloader, NUM_EPOCHS, LR, CRITERION, device='cuda')

    # evaluate the model on the test set
    evaluate_model(model, test_functaloader, CRITERION, 'Test', device='cuda')

    # visualize correct and incorrect classifications
    visualize_classifications(model, test_functaloader, inr, num_images=5, device='cuda', output_file='classification_results.png')

    # save model
    torch.save({'state_dict': model.state_dict()}, 'classifier.pth')
