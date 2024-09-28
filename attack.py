import tqdm
import torch
from torch import nn, optim
from utils import set_random_seeds, vec_to_img, get_fmnist_functa
import numpy as np
from SIREN import ModulatedSIREN
from classifier import WeightSpaceClassifier
import argparse

import evaluation_questions  # TODO LEFT: remove after generating plots


def attack_classifier(model, loader, criterion, linf_bound, num_pgd_steps=10, lr=0.01, device="cuda", 
                      return_preds=False, return_perturbations=False):
    """
    :param model: your trained classifier model
    :param loader: data loader for input to be perturbed
    :param criterion: The loss criteria you wish to maximize in attack
    :param linf_bound: L_inf norm bound for perturbations
    :param num_pgd_steps: Number of PGD steps to apply perturbations
    :param lr: learning rate for the perturbations
    :param device: Device to use for computation (cuda or cpu)
    :param return_preds: If True, return labels and predictions for confusion matrix generation
    :param return_perturbations: If True, return perturbations for visualization

    :return: Classification accuracy after attack, optionally predictions, true labels, and perturbations
    """
    model.eval()  # Model should be used in evaluation mode - we are not training any model weights.
    
    correct_predictions = 0

    if return_preds:  # TODO LEFT: remove the logic for return_preds before submitting
        true_labels = []
        predicted_labels = []

    if return_perturbations:  # TODO LEFT: remove the logic for return_perturbations before submitting
        perturbations = []

    prog_bar = tqdm.tqdm(loader, total=len(loader), leave=False)

    for vectors, labels in prog_bar:
        vectors, labels = vectors.to(device), labels.to(device)
        
        # initialize the perturbation vectors for current batch
        perts = torch.zeros_like(vectors, requires_grad=True)  # TODO (1): Allow gradient tracking        
        
        optimizer = optim.Adam([perts], lr=lr)  # TODO (2): Optimizer for the perturbations, not the model
        
        # Every step here is one PGD iteration (meaning, one attack optimization step) optimizing your perturbations.
        # After the loop below is over you'd have all fully-optimized perturbations for the current batch of vectors.
        for step in range(num_pgd_steps): 

            preds = model(vectors + perts)  # feed currently perturbed data into the model
            
            # TODO (3): Calculate loss (negate to maximize it)
            loss = -criterion(preds, labels)
            
            # Backpropagate and optimize the perturbations
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # TODO (4): Apply L inf norm bound projection, use 'torch.clamp' to ensure perturbations are within bounds
            perts.data = torch.clamp(perts.data, -linf_bound, linf_bound)

            eps = 1e-6  # Set a small tolerance for floating-point comparisons # TODO LEFT: check if its ok to do so
            assert perts.abs().max().item() <= linf_bound + eps  # If this assert fails, you have a mistake in TODO(4) 

            perts = perts.detach().requires_grad_()  # Reset gradient tracking - we don't want to track gradients for norm projection.

        # TODO (5): Final predictions for current batch after attack
        final_preds = torch.argmax(model(vectors + perts), dim=1)  # TODO LEFT: need softmax before argmax?
        correct_predictions += (final_preds == labels).sum().item()

        # Store predictions for confusion matrix
        if return_preds:
            true_labels.extend(labels.detach().cpu().numpy())
            predicted_labels.extend(final_preds.detach().cpu().numpy())

        if return_perturbations:
            perturbations.extend((perts).detach().cpu().numpy())

    # Return accuracy after attack
    total_samples = len(loader.dataset)
    accuracy = correct_predictions / total_samples * 100

    if return_preds and return_perturbations:
        return accuracy, true_labels, predicted_labels, perturbations
    elif return_preds:
        return accuracy, true_labels, predicted_labels
    elif return_perturbations:
        return accuracy, perturbations
    else:
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
    
    linf_bounds = [10**(-i) for i in range(3,7)] + [5*10**(-i) for i in range(3,7)]  # TODO LEFT: consider sorting the bounds
    
    linf_bounds.sort() # TODO LEFT: check if we can sort

    # define hyperparameters
    CRITERION = nn.CrossEntropyLoss()
    LR = 0.05  # TODO LEFT: continue to fine-tune for maximum accuracy reduction?

    accuracies = []
    
    for bound in linf_bounds:
        accuracy = attack_classifier(classifier, test_loader, CRITERION, linf_bound=bound, lr=LR, device=device)
        accuracies.append(accuracy)
        print(f"Test accuracy after attack with linf_bound={bound}: {accuracy:.2f}%")

    # TODO LEFT: remove this
    """
    learning_rates = [0.001, 0.005, 0.01, 0.02, 0.05]  # Different LRs to test

    for lr in learning_rates:
        for bound in linf_bounds:
            print(f"\nRunning attack with lr={lr} and linf_bound={bound}")
            accuracy = attack_classifier(classifier, test_loader, CRITERION, linf_bound=bound, lr=lr, device=device)
            print(f"Test accuracy after attack: {accuracy:.2f}%")
    """

    # --------- Generate results for evaluation questions ---------

    # Q6: Plot accuracies after attack
    evaluation_questions.plot_accuracies_after_attack(linf_bounds, accuracies)

    # Q7: Plot confusion matrix for adversarially attacked test set
    bounds_with_acc_20_to_80 = [bound for bound, accuracy in zip(linf_bounds, accuracies) if 20 <= accuracy <= 80]
    bound_for_q7 = min(bounds_with_acc_20_to_80)  # TODO LEFT: can choose another

    accuracy, true_labels, predicted_labels, perturbations = attack_classifier(classifier, test_loader, CRITERION, 
                                                                               linf_bound=bound_for_q7, lr=LR, 
                                                                               device=device, return_preds=True, 
                                                                               return_perturbations=True)
    class_names = [str(i) for i in range(10)]  # for Fashion MNIST classes
    evaluation_questions.plot_confusion_matrix(true_labels, predicted_labels, class_names, 
                                               set_name=f'Adversarial Test (epsilon={bound_for_q7})')

    # Q9: Visualize clean images, perturbations, and perturbed images for 3 classes (same bound as Q7)
    selected_classes = [0, 1, 2]  # TODO LEFT: can select others
    
    # Load Full INR - for visualization purposes
    inr = ModulatedSIREN(height=28, width=28, hidden_features=256, num_layers=10, modul_features=512)
    inr.load_state_dict(torch.load(f"{args.data_path}/modSiren.pth")['state_dict'])
    inr = inr.to(device)
    
    clean_vectors = [modul for modul, _ in test_loader.dataset]

    evaluation_questions.visualize_class_attack_results(clean_vectors, perturbations, 
                                                        true_labels, predicted_labels, class_names, inr, 
                                                        selected_classes, num_samples=1)
