import tqdm
import torch
from torch import nn, optim
from utils import set_random_seeds, vec_to_img, get_fmnist_functa
import numpy as np
from SIREN import ModulatedSIREN
from classifier import VanillaClassifier
import argparse


def attack_classifier(model, loader, criterion, linf_bound, num_pgd_steps = 10, device = "cuda"):
    """
    :param model: your trained classifier model
    :param loader: data loader for input to be perturbed
    :param criterion: The loss criteria you wish to maximize in attack
    """
    model.eval() #Model should be used in evaluation mode - we are not training any model weights.
 
    
    
    success = []  # TODO LEFT: what to put in this list? success of the model despite the attack? see piazza
    prog_bar = tqdm.tqdm(loader, total=len(loader))
    for vectors, labels in prog_bar:
   
        vectors, labels = vectors.to(device), labels.to(device)
        
        perts = torch.zeros_like(vectors) #initialize the perturbation vectors for current iteration
        
        ''' TODO (1): Your perts tensor currently will not be optimized since torch wasn't instructed to track gradients for it - make torch track its gradients. '''
        
        
        ''' TODO (2): Initialize your optimizer, you might need to finetune the learn-rate.
        What should be the set of parameters the optimizer will be changing? Hint: NOT model.parameters()!
        '''
        optimizer = None
        
        
        '''Every step here is one PGD iteration (meaning, one attack optimization step) optimizing your perturbations.
        After the loop below is over you'd have all fully-optimized perturbations for the current batch of vectors.'''
        for step in range(num_pgd_steps): 
           
            preds = model(vectors + perts) #feed currently perturbed data into the model
            loss = criterion(preds, labels) ''' TODO (3):  What's written in this line for the loss is almost correct. Change the code to MAXIMIZE the loss'''
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ''' TODO (4): Perform needed L_inf norm bound projection. The 'torch.clamp' function could be useful.'''
            
            assert perts.abs().max().item() <= linf_bound #If this assert fails, you have a mistake in TODO(4) 
            perts = perts.detach().requires_grad() #Reset gradient tracking - we don't want to track gradients for norm projection.
            # TODO LEFT: need to change requires_grad to requires_grad_(True) ? see piazza
            
        ''' TODO (5): Accumulate predictions and labels to compute final accuracy for the attacked classifier.
        You can compute final predictions by taking the argmax over the softmax of predictions.'''
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize an adversarial attack over a pre-trained weight-space classifier')
    parser.add_argument('-p', '--data-path', type=str, default='/datasets/functaset',
                        help='The path to INR dataset (a.k.a functaset)')
    parser.add_argument('-b', '--batch-size', type=int, default=128,
                        help='The path to INR dataset (a.k.a functaset)')
    parser.add_argument('-c', '--cpu', action='store_true', help = "If set, use cpu and not cuda")
    parser.add_argument('-m', '--model-path', type=str, help="Path to your pretrained classifier model weights")
    # add any other parameters you may need here
    args = parser.parse_args()
    
    # Set random seed.
    set_random_seeds(0)
    device = 'cpu' if args.cpu else 'cuda:0'
    
       
    # Instantiate Classifier Model and load weights
    classifier = VanillaClassifier(in_features=512, num_classes=10).to(device)
    classifier.load_state_dict(torch.load(args.model_path)['state_dict'])
    
    
    linf_bounds = [10**(-i) for i in range(3,7)] + [5*10**(-i) for i in range(3,7)]  # TODO LEFT: consider sorting the bounds
    
    for bound in linf_bounds:
        pass #call the attack_classifier function for every bound