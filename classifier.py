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
from evaluation_questions import plot_confusion_matrix_for_model


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


def report_classification_accuracy(model, criterion, train_loader, val_loader, test_loader, device='cuda'):
    """
    Reports classification accuracy on training, validation, and test sets.
    """
    print("Evaluating Training Set...")
    evaluate_model(model, train_loader, criterion, 'Training', device)
    
    print("Evaluating Validation Set...")
    evaluate_model(model, val_loader, criterion, 'Validation', device)
    
    print("Evaluating Test Set...")
    evaluate_model(model, test_loader, criterion, 'Test', device)


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
    inr = ModulatedSIREN(height=28, width=28, hidden_features=256, num_layers=10, modul_features=512, device=device)
    inr.load_state_dict(torch.load(f"{args.data_path}/modSiren.pth", map_location=device)['state_dict'])
    inr = inr.to(device)
    
    # TODO: Implement your training and evaluation loops here. We recommend you also save classifier weights for next parts
    
    # --------- Implementation ---------

    # define hyperparameters
    DROPOUT = 0.3  # for 0.25 got 88.73% test
    BATCHNORM = True  # w/o got 85% test
    NUM_EPOCHS = 50
    LR = 0.001  # with 0.0005 got 88.30% test, with 0.0001 got 87.76% test
    CRITERION = nn.CrossEntropyLoss()  # with label_smoothing=0.1 and WEIGHT_DECAY=7e-5 got 88.64%
    PATIENCE = 10
    WEIGHT_DECAY = 5e-5  # with 1e-4 got 88.86% test, with 7e-5 got 88.88%
    LR_SCHEDULER = True

    # initialize model
    model = WeightSpaceClassifier(use_batchnorm=BATCHNORM, dropout_prob=DROPOUT).to(device)

    # train model (validation set is evaluated at the end of each epoch)
    train_model(model, train_functaloader, val_functaloader, NUM_EPOCHS, LR, CRITERION, PATIENCE, WEIGHT_DECAY, LR_SCHEDULER, device=device)

    # evaluate the model on the test set
    evaluate_model(model, test_functaloader, CRITERION, 'Test', device=device)


    # --------- generate results for evaluation questions ---------
    
    # Q1: report classification accuracy for train, validation and test sets
    report_classification_accuracy(model, CRITERION, train_functaloader, val_functaloader, test_functaloader, device)
    
    # Q2: plot confusion matrix for the each of the train, validation and test sets (3 plots in total)
    class_names = [str(i) for i in range(10)]  # for Fashion MNIST classes
    for loader, set_name in zip([train_functaloader, val_functaloader, test_functaloader], ['Train', 'Validation', 'Test']):
        plot_confusion_matrix_for_model(model, loader, class_names, set_name)

    # save model
    torch.save({'state_dict': model.state_dict()}, 'classifier.pth')
