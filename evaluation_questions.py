import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from classifier import evaluate_model


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


def plot_confusion_matrix(model, loader, class_names, set_name, device='cuda'):
    """
    Plots and saves confusion matrix for a given dataset.
    
    :param model: Trained model
    :param loader: DataLoader for dataset
    :param class_names: List of class names (e.g., ['0', '1', ..., '9'] for Fashion MNIST)
    :param set_name: Name of the dataset (e.g., 'Train', 'Validation', 'Test') for the plot title
    :param device: Device ('cuda' or 'cpu') to run inference
    """
    model.eval()
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, 1)
            true_labels.extend(labels.cpu().detach().numpy())
            predicted_labels.extend(predicted.detach().cpu().numpy())
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {set_name} Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save the plot to a file
    output_file = set_name + '_confusion_matrix.png'
    plt.savefig(output_file)
    plt.close()  # Close the figure to prevent it from displaying
    
    print(f"Confusion matrix for {set_name} set saved as {output_file}")
