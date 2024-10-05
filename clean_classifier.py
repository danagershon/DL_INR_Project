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


class WeightSpaceClassifier(nn.Module):
    """
    Our classifer implementation
    """
    
    def __init__(self, in_features=512, num_classes=10, use_batchnorm=True, dropout_prob=0.2):
        """
        :param in_features: input_dimension (size of modulation vector)
        :param num_classes: number of classes (output dimension). 10 classes, for each type of clothing.
        :param use_batchnorm: whether to add a batchnom layer
        :param dropout_prob: dropout probability (if 0, means no dropout)
        """
        super(WeightSpaceClassifier, self).__init__()

        layers = [
            # First hidden layer
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity(),

            # Second hidden layer
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity(),

            nn.Linear(128, num_classes)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

def train_model(model, 
                train_loader, 
                val_loader, 
                num_epochs=20, 
                learning_rate=0.001, 
                criterion=nn.CrossEntropyLoss(), 
                patience=5,
                weight_decay=5e-5,
                lr_scheduler=True,
                device='cuda'):
    """
    Train the WeightSpaceClassifier model with early stopping.

    :param model: The classifier model.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param num_epochs: Number of epochs to train.
    :param learning_rate: Learning rate for the optimizer.
    :param criterion: the loss function.
    :param patience: Number of epochs to wait for improvement before stopping early.
    :param weight_decay: L2 regularization weight for model weights (for optimizer).
    :param lr_scheduler: add learning rate scheduler.
    :param device: Device to train on ('cuda' or 'cpu').

    :return: Trained model.
    """
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # weight_decay adds L2 regularization
    
    if lr_scheduler:
        # Learning rate scheduler (reduces LR when validation accuracy plateaus)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)

    # Initialize early stopping variables
    best_val_acc = float('-inf')
    best_model_weights = None
    patience_counter = 0

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

            # Update model parameters
            optimizer.step()

            # Accumulate epoch loss and correct predictions
            epoch_loss += loss.item() * inputs.size(0)  # multiply by batch size
            predicted = torch.argmax(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for the epoch
        total_samples = len(train_loader.dataset)
        avg_epoch_loss = epoch_loss / total_samples
        epoch_accuracy = (correct_predictions / total_samples) * 100

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Evaluate on the validation set after each epoch
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, 'Validation', device)
        
        if lr_scheduler:
            # Step the learning rate scheduler based on validation accuracy
            scheduler.step(val_accuracy)

        # Early stopping check: If validation loss improves, reset patience and save best model weights
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_weights = model.state_dict().copy()  # Save the current best model
            patience_counter = 0  # Reset the patience counter
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")

        # Stop training if patience is exceeded
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load the best model weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print("Restored best model weights.")

    return model


def evaluate_model(model, loader, criterion, type, device='cuda'):
    """
    Evaluate the model (on the validation/test set).

    :param model: The classifier model.
    :param loader: DataLoader for validation/test data.
    :param criterion: the loss function
    :param type: 'Validation' or 'Test' (for printing purposes)
    :param device: Device to evaluate on ('cuda' or 'cpu').

    :return: validation/test loss and accuracy
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0

    # Use no_grad to disable gradient calculation during evaluation
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute batch loss
            loss = criterion(outputs, labels)

            # Accumulate validation loss and correct predictions
            total_loss += loss.item() * inputs.size(0)
            predicted = torch.argmax(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

    # Calculate average validation loss and accuracy
    total_samples = len(loader.dataset)
    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples * 100

    print(f"{type} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy


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
        
    # --------- Implementation ---------

    # define hyperparameters
    DROPOUT = 0.3
    BATCHNORM = True
    NUM_EPOCHS = 50
    LR = 0.001
    CRITERION = nn.CrossEntropyLoss()
    PATIENCE = 10
    WEIGHT_DECAY = 5e-5
    LR_SCHEDULER = True

    # initialize model
    model = WeightSpaceClassifier(use_batchnorm=BATCHNORM, dropout_prob=DROPOUT).to(device)

    # train model (validation set is evaluated at the end of each epoch)
    train_model(model, train_functaloader, val_functaloader, NUM_EPOCHS, LR, CRITERION, PATIENCE, WEIGHT_DECAY, LR_SCHEDULER, device=device)

    # evaluate the model on the test set
    evaluate_model(model, test_functaloader, CRITERION, 'Test', device=device)
