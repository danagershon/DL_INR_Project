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
from classifier import WeightSpaceClassifier, evaluate_model, report_classification_accuracy


def attack_classifier(model, inputs, labels, criterion, linf_bound, num_pgd_steps=10, lr=0.01):
    """
    :param model: your trained classifier model
    :param criterion: The loss criteria you wish to maximize in attack
    :param linf_bound: L_inf norm bound for perturbations
    :param num_pgd_steps: Number of PGD steps to apply perturbations
    :param lr: learning rate for the perturbations
    :param device: Device to use for computation (cuda or cpu)

    :return: perturbations
    """
    model.eval()  # Model should be used in evaluation mode - we are not training any model weights.

    # initialize the perturbation inputs for current batch
    perts = torch.zeros_like(inputs, requires_grad=True)  # TODO (1): Allow gradient tracking        
    
    optimizer = optim.Adam([perts], lr=lr)  # TODO (2): Optimizer for the perturbations, not the model
    
    # Every step here is one PGD iteration (meaning, one attack optimization step) optimizing your perturbations.
    # After the loop below is over you'd have all fully-optimized perturbations for the current batch of inputs.
    for step in range(num_pgd_steps): 
        preds = model(inputs + perts)  # feed currently perturbed data into the model
        
        # TODO (3): Calculate loss (negate to maximize it)
        loss = -criterion(preds, labels)
        
        # Backpropagate and optimize the perturbations
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # TODO (4): Apply L inf norm bound projection, use 'torch.clamp' to ensure perturbations are within bounds
        perts.data = torch.clamp(perts.data, -linf_bound + 1e-9, linf_bound - 1e-9)
        
        assert perts.abs().max().item() <= linf_bound  # If this assert fails, you have a mistake in TODO(4) 
        
        perts = perts.detach().requires_grad_()  # Reset gradient tracking - we don't want to track gradients for norm projection.

    return perts.detach()


def train_model_AT(model, 
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
    Adversarial training of the WeightSpaceClassifier model with early stopping.

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
        total_clean_loss = 0.0
        total_adv_loss = 0.0
        correct_predictions_clean = 0
        correct_predictions_adv = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # ---- Training on original samples ----
            optimizer.zero_grad()
            outputs_clean = model(inputs)
            clean_loss = criterion(outputs_clean, labels)
            clean_loss.backward()
            optimizer.step()

            # Add clean loss to total loss
            total_clean_loss += clean_loss.item() * inputs.size(0)  # multiply by batch size

            # Calculate correct predictions for clean data
            predicted_clean = torch.argmax(outputs_clean, 1)
            correct_predictions_clean += (predicted_clean == labels).sum().item()
            
            # ---- Bonus: Training on adversarial samples ----
            perturbations = attack_classifier(model, inputs, labels, criterion=nn.CrossEntropyLoss(), 
                                              linf_bound=1e-4, lr=0.01, num_pgd_steps=10)
            perturbed_inputs = inputs + perturbations
            optimizer.zero_grad()
            outputs_adv = model(perturbed_inputs)
            adv_loss = criterion(outputs_adv, labels)
            adv_loss.backward()
            optimizer.step()

            # Add adversarial loss to total loss
            total_adv_loss += adv_loss.item() * inputs.size(0)  # multiply by batch size

            # Calculate correct predictions for adversarial data
            predicted_adv = torch.argmax(outputs_adv, 1)
            correct_predictions_adv += (predicted_adv == labels).sum().item()

        # Calculate average loss and accuracy for both clean and adversarial data
        total_samples = len(train_loader.dataset)
        avg_clean_loss = total_clean_loss / total_samples
        avg_adv_loss = total_adv_loss / total_samples
        avg_epoch_loss = (total_clean_loss + total_adv_loss) / (2 * total_samples)
        epoch_accuracy_clean = (correct_predictions_clean / total_samples) * 100
        epoch_accuracy_adv = (correct_predictions_adv / total_samples) * 100

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Clean Loss: {avg_clean_loss:.4f}, Adv Loss: {avg_adv_loss:.4f}, Avg Loss: {avg_epoch_loss:.4f}")
        print(f"Accuracy (Clean): {epoch_accuracy_clean:.2f}%, Accuracy (Adv): {epoch_accuracy_adv:.2f}%")

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
        
    # --------- Implementation ---------

    # define hyperparameters (same as the original classifier)
    DROPOUT = 0.3
    BATCHNORM = True 
    NUM_EPOCHS = 50
    LR = 0.001
    CRITERION = nn.CrossEntropyLoss()
    PATIENCE = 20
    WEIGHT_DECAY = 5e-5
    LR_SCHEDULER = True

    # initialize model
    model = WeightSpaceClassifier(use_batchnorm=BATCHNORM, dropout_prob=DROPOUT).to(device)

    # train model (validation set is evaluated at the end of each epoch)
    train_model_AT(model, train_functaloader, val_functaloader, NUM_EPOCHS, LR, CRITERION, PATIENCE, WEIGHT_DECAY, LR_SCHEDULER, device=device)

    # evaluate the model on the test set
    evaluate_model(model, test_functaloader, CRITERION, 'Test', device=device)

    # --------- generate results for evaluation questions ---------
    
    # Q1: report classification accuracy for train, validation and test sets
    report_classification_accuracy(model, CRITERION, train_functaloader, val_functaloader, test_functaloader, device)

    # save model
    torch.save({'state_dict': model.state_dict()}, 'classifier.pth')
