import tqdm
import torch
from torch import nn, optim
from utils import set_random_seeds, vec_to_img, get_fmnist_functa
import numpy as np
from SIREN import ModulatedSIREN
from classifier import WeightSpaceClassifier
import argparse


"""
Execution instructions:

For the original model: After executing classifier.py:
attack.py --model-path classifier.pth

For the adversarially trained model: After executing classifier_bunos_AT.py:
attack.py --model-path classifier_AT.pth
"""


def attack_classifier(model, loader, criterion, linf_bound, num_pgd_steps=10, lr=0.01, device="cuda"):
    """
    :param model: your trained classifier model
    :param loader: data loader for input to be perturbed
    :param criterion: The loss criteria you wish to maximize in attack
    :param linf_bound: L_inf norm bound for perturbations
    :param num_pgd_steps: Number of PGD steps to apply perturbations
    :param lr: learning rate for the perturbations
    :param device: Device to use for computation (cuda or cpu)

    :return: Classification accuracy after attack
    """
    model.eval()  # Model should be used in evaluation mode - we are not training any model weights.
    
    correct_predictions = 0

    prog_bar = tqdm.tqdm(loader, total=len(loader), leave=False)

    for vectors, labels in prog_bar:
        vectors, labels = vectors.to(device), labels.to(device)
        
        # initialize the perturbation vectors for current batch
        perts = torch.zeros_like(vectors, requires_grad=True)  # Allow gradient tracking        
        
        optimizer = optim.Adam([perts], lr=lr)  # Optimizer for the perturbations, not the model
        
        # Every step here is one PGD iteration (meaning, one attack optimization step) optimizing your perturbations.
        # After the loop below is over you'd have all fully-optimized perturbations for the current batch of vectors.
        for step in range(num_pgd_steps): 

            preds = model(vectors + perts)  # feed currently perturbed data into the model
            
            # Calculate loss (negate to maximize it)
            loss = -criterion(preds, labels)
            
            # Backpropagate and optimize the perturbations
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Apply L inf norm bound projection, use 'torch.clamp' to ensure perturbations are within bounds
            perts.data = torch.clamp(perts.data, -linf_bound + 1e-9, linf_bound - 1e-9)

            assert perts.abs().max().item() <= linf_bound  # If this assert fails, you have a mistake in TODO(4) 

            perts = perts.detach().requires_grad_()  # Reset gradient tracking - we don't want to track gradients for norm projection.

        # Final predictions for current batch after attack
        final_preds = torch.argmax(model(vectors + perts), dim=1)
        correct_predictions += (final_preds == labels).sum().item()

    # Return accuracy after attack
    total_samples = len(loader.dataset)
    accuracy = correct_predictions / total_samples * 100

    return accuracy
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize an adversarial attack over a pre-trained weight-space classifier')
    parser.add_argument('-p', '--data-path', type=str, default='/datasets/functaset',
                        help='The path to INR dataset (a.k.a functaset)')
    parser.add_argument('-b', '--batch-size', type=int, default=128,
                        help='batch size for the data loader')
    parser.add_argument('-c', '--cpu', action='store_true', help = "If set, use cpu and not cuda")
    parser.add_argument('-m', '--model-path', type=str, default='classifier.pth',
                        help="Path to your pretrained classifier model weights")

    # add any other parameters you may need here
    args = parser.parse_args()
    
    # Set random seed.
    set_random_seeds(0)
    device = 'cpu' if args.cpu else 'cuda:0'

    # Load test data
    test_loader = get_fmnist_functa(data_dir=f"{args.data_path}/fmnist_test.pkl", mode='test', batch_size=args.batch_size, num_workers=2)
    
    # Instantiate Classifier Model and load weights
    classifier = WeightSpaceClassifier(in_features=512, num_classes=10).to(device)
    classifier.load_state_dict(torch.load(args.model_path)['state_dict'])
    
    linf_bounds = [10**(-i) for i in range(3,7)] + [5*10**(-i) for i in range(3,7)]
    
    linf_bounds.sort()

    # define hyperparameters
    CRITERION = nn.CrossEntropyLoss()
    LR = 0.05

    accuracies = []
    
    for bound in linf_bounds:
        accuracy = attack_classifier(classifier, test_loader, CRITERION, linf_bound=bound, lr=LR, device=device)
        accuracies.append(accuracy)
        print(f"Test accuracy after attack with linf_bound={bound}: {accuracy:.2f}%")
